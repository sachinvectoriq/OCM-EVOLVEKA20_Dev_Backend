"""
Agentic RAG Workflow using Microsoft Agent Framework.

This workflow implements an iterative retrieval-augmented generation pattern with:
- HyDE (Hypothetical Document Embedding) query rewriting for semantic search
- Intelligent reflection and result vetting with LLM-based quality review
- Conditional routing with up to 3 search iterations
- Answer generation with proper citation tracking
- Class-based architecture with dependency injection for testability
"""

from typing import List
from agent_framework import (
    WorkflowBuilder,
    Workflow,
    WorkflowContext,
)
from agent_framework._workflows._function_executor import FunctionExecutor

from app.models.chat import AgenticRAGState, RetrievedDocument
from app.models import WorkflowOptions
from app.core.settings import Settings
from app.core.logger import Logger
from app.core.language import Language
from app.agents.query_rewriter import QueryRewriter
from app.agents.answer_generator import AnswerGenerator
from app.agents.reflection_agent import ReflectionAgent
from app.services.search_service import ISearchService
from app.utils.citation_tracker import CitationTracker
from app.prompts.templates import AnswerGeneratorPrompts
from app.prompts.localized import (
    get_prompts,
    FALLBACK_NO_RELEVANT_CONTENT,
    FALLBACK_LIMITED_RESULTS,
    SEARCHED_WITHOUT_FILTERS_NOTE,
)


class AgenticRAGWorkflow:
    """
    Agentic RAG Workflow orchestrating iterative retrieval with reflection.
    
    This workflow implements a multi-agent pattern where:
    1. Query Rewriter Agent generates HyDE search queries
    2. Search Executor retrieves documents from Azure AI Search
    3. Reflection Agent reviews results and decides whether to continue or finalize
    4. Answer Generator Agent synthesizes grounded answers with citations
    
    The workflow uses conditional routing to enable up to 3 search iterations,
    with intelligent decision-making at each step to balance thoroughness with efficiency.
    
    Flow:
        Initialize → Search → Reflection → [retry: Search | finalize: Answer] → Complete
        
    Decision routing:
    - "retry" + attempts < 3 → Loop back to Search with refined query
    - "finalize" OR attempts = 3 → Proceed to Answer generation
    """
    
    def __init__(
        self,
        settings: Settings,
        logger: Logger,
        workflow_options: WorkflowOptions,
        search_service,
        citation_tracker: CitationTracker,
        query_rewriter: QueryRewriter,
        answer_generator: AnswerGenerator,
        reflection_agent: ReflectionAgent,
    ):
        """Initialize workflow with all required dependencies.

        ``search_service`` may be either:
        - an ``ISearchService`` instance (legacy single-language behaviour), or
        - a callable ``Callable[[str], ISearchService]`` that resolves the
          right service for a given language code (preferred). When a state
          has ``language="fr-ca"``, the workflow asks the resolver for the
          French-Canadian search service.
        """
        self.settings = settings
        self.logger = logger
        self.workflow_options = workflow_options
        self.search_service = search_service  # ISearchService or resolver callable
        self.citation_tracker = citation_tracker
        self.query_rewriter = query_rewriter
        self.answer_generator = answer_generator
        self.reflection_agent = reflection_agent

    def _resolve_search_service(self, language: str) -> ISearchService:
        """Return the right ISearchService for the given language code."""
        if callable(self.search_service) and not isinstance(self.search_service, ISearchService):
            return self.search_service(language)
        return self.search_service
    
    async def search_executor(self, state: AgenticRAGState, ctx: WorkflowContext[AgenticRAGState]) -> None:
        """Execute search iteration with HyDE query rewriting."""
        state.current_attempt += 1
        self.logger.info(f"[Search] EXECUTOR CALLED - Attempt {state.current_attempt}/{state.max_attempts}")
        
        # Generate HyDE query if enabled, otherwise use original query
        if (self.workflow_options.enable_query_rewriting and state.current_attempt > 1):  # Only rewrite on the first attempt to preserve user intent in retries
            search_query = await self.query_rewriter.generate_hyde_search_query(
                user_query=state.query,
                search_history=state.search_history,
                previous_reviews=state.previous_reviews,
                language=state.language,
                conversation_history=state.conversation_history,
            )
        else:
            search_query = state.query
            self.logger.info("[Search] Query rewriting disabled, using original query")
                
        try:
            # Resolve language-bound search service
            search_service = self._resolve_search_service(state.language)
            # Execute search with filters if they exist, otherwise search without filters. Exclude already processed content_ids to maximize unique results each iteration.
            results = await search_service.search_async(
                query=search_query,
                search_mode="hybrid",
                top_k=10,
                filters=state.filters,
                exclude_ids=list(state.processed_content_ids)
            )
            
            # If no results and filters were applied, retry once without filters to broaden search
            if not results and state.filters:
                self.logger.info(
                    f"[Search] No results with filters {state.filters}, retrying without filters"
                )
                results = await search_service.search_async(
                    query=search_query,
                    search_mode="hybrid",
                    top_k=10,
                    filters=None,
                    exclude_ids=list(state.processed_content_ids)
                )                            
                # Mark that we've bypassed filters to track in state and thought process
                state.filters = None
                state.searched_without_filters = True

            state.current_results = results
            state.search_history.append({
                "query": search_query,
                "results_count": len(results),
                "attempt": state.current_attempt
            })
            
            state.thought_process.append({
                "step": "retrieve",
                "details": {
                    "attempt": f"{state.current_attempt} out of {state.max_attempts}",
                    "user_query": state.query,
                    "generated_search_query": search_query,
                    "applied_filters": ({k: v for k, v in state.filters.items() if v} or None) if state.filters else None,
                    "searched_without_filters": state.searched_without_filters,
                    "results_summary": [
                        {
                            "content_id": result.content_id,
                            "document_id": result.document_id,
                            "title": result.title,
                            "score": result.score,
                            "reranker_score": result.reranker_score,
                            "content": result.content
                        }
                        for result in results
                    ]
                }
            })
            
            self.logger.info(f"[Search] Found {len(results)} results")
            
        except Exception as e:
            self.logger.error(f"[Search] Failed: {e}")
            state.current_results = []
        
        # Send state to next executor (reflection)
        await ctx.send_message(state)


    
    
    async def reflection_executor(self, state: AgenticRAGState, ctx: WorkflowContext[AgenticRAGState]) -> None:
        """Review search results and decide whether to continue or finalize."""
        self.logger.info(f"[Reflection] EXECUTOR CALLED - Reviewing {len(state.current_results)} results")
        
        
        
        if not state.current_results:
            if state.current_attempt < state.max_attempts:
                self.logger.info(f"[Reflection] No results on attempt {state.current_attempt}. Retrying...")
                state.decision = "search"
            else:
                self.logger.warning(f"[Reflection] No results after {state.current_attempt} attempts. Finalizing.")
                state.decision = "finalize"
            await ctx.send_message(state)
            return

        if state.current_attempt == 1 and len(state.current_results) >= 3:


            high_confidence_results = [
                 r for r in state.current_results
                 if r.reranker_score is not None and r.reranker_score >= 2.4
            ]
              
            if len(high_confidence_results) >= 2:
                self.logger.info("[Reflection] Fast finalize triggered (skipping LLM reflection)")
                state.vetted_results = high_confidence_results
                state.decision = "finalize"
                await ctx.send_message(state)
                return

        try:
            # Use ReflectionAgent to review results
            decision, new_vetted, discarded, llm_original_decision = await self.reflection_agent.review_search_results(
                user_query=state.query,
                current_results=state.current_results,
                vetted_results=state.vetted_results,
                search_history=state.search_history,
                max_attempts=state.max_attempts,
                current_attempt=state.current_attempt,
                language=state.language,
            )
            
            # Update state with reviewed results
            state.vetted_results.extend(new_vetted)
            state.discarded_results.extend(discarded)
            state.previous_reviews.append(decision.thought_process)
            
            # Store final decision (after smart retry logic)
            final_decision = decision.decision
            state.decisions.append(final_decision)
            
            # Mark all current results as processed (by content_id to track unique chunks)
            for doc in state.current_results:
                state.processed_content_ids.add(doc.content_id)
            
            # Calculate metrics for logging
            current_count = len(state.current_results)
            valid_count = len(new_vetted)
            valid_percentage = valid_count / current_count if current_count > 0 else 0
            
            # Log thought process (matching archive-chat-agent pattern)
            state.thought_process.append({
                "step": "review",
                "details": {
                    "attempt": f"{state.current_attempt} out of {state.max_attempts}",
                    "review_thought_process": decision.thought_process,
                    "valid_results": [
                        {
                            "content_id": doc.content_id,
                            "document_id": doc.document_id,
                            "title": doc.title,
                            "score": doc.score,
                            "reranker_score": doc.reranker_score,
                            "content": doc.content
                        }
                        for doc in new_vetted
                    ],
                    "invalid_results": [
                        {
                            "content_id": doc.content_id,
                            "document_id": doc.document_id,
                            "title": doc.title,
                            "score": doc.score,
                            "reranker_score": doc.reranker_score,
                            "content": doc.content
                        }
                        for doc in discarded
                    ],
                    "llm_decision": llm_original_decision,
                    "final_decision": final_decision,
                    "decision_override": final_decision != llm_original_decision,
                    "valid_count": valid_count,
                    "invalid_count": len(discarded),
                    "valid_percentage": f"{valid_percentage:.0%}",
                    "total_vetted": len(state.vetted_results)
                }
            })
            
            # Clear current results for next iteration
            state.current_results = []
            
            # Set decision for routing
            if decision.decision == "retry" and state.current_attempt < state.max_attempts:
                state.decision = "search"
                self.logger.info(f"[Reflection] ROUTING DECISION: search (continue iteration {state.current_attempt}/{state.max_attempts})")
            elif not state.vetted_results and not state.searched_without_filters and state.filters is not None:
                self._retry_without_filters(state)
            else:
                state.decision = "finalize"
                self.logger.info(f"[Reflection] ROUTING DECISION: finalize with {len(state.vetted_results)} vetted results")
            
        except Exception as e:
            self.logger.error(f"[Reflection] Failed: {e}")
            state.decision = "finalize"
        
        # Send state to next executor based on decision
        await ctx.send_message(state)
    
    async def answer_generator_executor(self, state: AgenticRAGState, ctx: WorkflowContext[AgenticRAGState]) -> None:
        """Generate final answer from vetted results."""
        self.logger.info(f"[AnswerGenerator] EXECUTOR CALLED - Generating from {len(state.vetted_results)} vetted results")
        
        try:            
            vetted_results_formatted = ""
            for i, doc in enumerate(state.vetted_results, 1):
                result_parts = [
                    f"\nResult #{i}",
                    "=" * 80,
                    f"Citation Token: ref-{i}",
                    f"Content ID: {doc.content_id}",
                    f"Document ID: {doc.document_id}",
                    f"Title: {doc.title}",
                    f"Source: {doc.source}",
                    f"Page Number: {doc.page_number if doc.page_number else 'N/A'}",
                    "\n<Start Content>",
                    "-" * 80,
                    doc.content,
                    "-" * 80,
                    "<End Content>"
                ]
                vetted_results_formatted += "\n".join(result_parts)

            # Build the answer prompt using the language-specific builder
            prompts = get_prompts(state.language)
            generated_answer_prompt = prompts.answer_prompt_builder(
                state.query, vetted_results_formatted
            )

            # Generate answer using answer_generator with the answer prompt that includes citation instructions and formatted vetted results
            generated_answer = await self.answer_generator.generate_answer(
                query=state.query,
                documents=state.vetted_results,
                generated_answer_prompt=generated_answer_prompt,
                conversation_history=state.conversation_history,
                language=state.language,
            )

            state.answer = generated_answer.answer_text
            state.citations = generated_answer.citations or []
            # Detect fallback answers (do NOT retry those) - check both EN and FR phrases
            lang = Language.from_value(state.language)
            fallback_phrases = [
                FALLBACK_NO_RELEVANT_CONTENT[Language.ENGLISH],
                FALLBACK_LIMITED_RESULTS[Language.ENGLISH],
                FALLBACK_NO_RELEVANT_CONTENT[Language.FRENCH_CANADIAN],
                FALLBACK_LIMITED_RESULTS[Language.FRENCH_CANADIAN],
            ]
            is_fallback = any(
                phrase in state.answer
                for phrase in fallback_phrases
            )
            if (
                state.vetted_results
                and not state.citations
                and not state.answer_retry_attempted
                and not is_fallback
            ):
                self.logger.warning(
                    "[AnswerGenerator] No citations returned for non-fallback answer. Retrying once."
                )
                state.answer_retry_attempted = True

                generated_answer = await self.answer_generator.generate_answer(
                    query=state.query,
                    documents=state.vetted_results,
                    generated_answer_prompt=generated_answer_prompt,
                    conversation_history=state.conversation_history,
                    language=state.language,
                )


                state.answer = generated_answer.answer_text
                state.citations = generated_answer.citations or []

            # If we had to bypass filters to get results, add a note to the answer to indicate this to the user for transparency
            if state.searched_without_filters:
                state.answer += SEARCHED_WITHOUT_FILTERS_NOTE[lang]

            state.thought_process.append({
                "step": "response",
                "details": {
                    "final_answer": state.answer,
                    "searched_without_filters": state.searched_without_filters,
                    "citations_count": len(state.citations),
                    "cited_documents": state.citations
                }
            })
            
            self.logger.info(f"[AnswerGenerator] Complete with {len(state.citations or [])} citations")
            
        except Exception as e:
            self.logger.error(f"[AnswerGenerator] Failed: {e}")
            state.answer = f"I encountered an error generating the final answer. Error: {str(e)}. Please try rephrasing your question."
            state.citations = []
            
            state.thought_process.append({
                "step": "response",
                "details": {
                    "final_answer": state.answer,
                    "error": str(e)
                }
            })
        
        # Yield final output
        await ctx.yield_output(state)  # type: ignore[attr-defined]
    
    def _retry_without_filters(self, state: AgenticRAGState) -> None:
        """Reset state to restart the full agentic loop without filters.

        Called when filters were applied throughout all search iterations but
        produced no vetted results. Clears filters and resets loop counters so
        the workflow can broaden the search on the next pass.
        """
        assert state.filters is not None
        cleared_filters = {k: v for k, v in state.filters.items()}
        state.decision = "retry_no_filters"
        state.filters = None
        state.searched_without_filters = True
        state.current_attempt = 0
        state.processed_content_ids = set()
        self.logger.info(
            "[Reflection] ROUTING DECISION: retry_no_filters "
            "(no vetted results from filtered search, restarting without filters)"
        )
        state.thought_process.append({
            "step": "broaden_search",
            "details": {
                "reason": "No vetted results returned after all search iterations with filters applied. Restarting workflow without filters.",
                "cleared_filters": cleared_filters,
            }
        })

    def should_finalize(self):
        """Condition for routing to the answer generator.

        Returns True when the reflection executor has set decision to "finalize".
        """
        def condition(message) -> bool:
            if hasattr(message, 'decision') and hasattr(message, 'current_attempt') and hasattr(message, 'max_attempts'):
                result = message.decision == "finalize"
                self.logger.info(f"[Condition] should_finalize={result}, decision={message.decision}, attempt={message.current_attempt}/{message.max_attempts}")
                return result
            return False
        return condition

    def should_search(self):
        """Condition for routing back to the search executor.

        Returns True for both normal iteration (decision == "search") and the
        filter-retry restart (decision == "retry_no_filters"), collapsing both
        into a single reflection→search edge as required by the framework.
        """
        def condition(message) -> bool:
            if hasattr(message, 'decision'):
                result = message.decision in ("search", "retry_no_filters")
                if result:
                    self.logger.info(f"[Condition] should_search=True (decision={message.decision})")
                return result
            return False
        return condition
    
    def build_workflow(self) -> Workflow:
        """
        Build and return the configured workflow.
        
        Returns:
            Configured workflow with conditional routing
        """
        self.logger.info("Building Agentic RAG Workflow...")
        
        # Create function executors from instance methods
        search_exec = FunctionExecutor(self.search_executor, id="search")
        reflection_exec = FunctionExecutor(self.reflection_executor, id="reflection")
        answer_exec = FunctionExecutor(self.answer_generator_executor, id="answer_generator")
        
        # Build workflow with conditional routing
        workflow = (
            WorkflowBuilder()
            .set_start_executor(search_exec)
            .add_edge(search_exec, reflection_exec)  # After search, always go to reflection
            # After reflection, conditionally route based on decision
            .add_edge(reflection_exec, search_exec, condition=self.should_search())        # Loop back if decision == "search" or "retry_no_filters"
            .add_edge(reflection_exec, answer_exec, condition=self.should_finalize())      # Finalize if decision == "finalize"
            .build()
        )

        self.logger.info("Agentic RAG workflow built: search → reflection → [search | retry_no_filters | answer_generator]")
        return workflow
