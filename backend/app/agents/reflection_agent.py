"""
Reflection Agent for evaluating search results and making continuation decisions.
Implements the review/reflection pattern from the reference architecture.
Uses MAF ChatAgent for LLM interactions.
"""
from typing import List, Dict, Any, Tuple, Literal
from azure.identity import DefaultAzureCredential

from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient
from opentelemetry import trace

from app.core.settings import Settings
from app.core.logger import Logger
from app.core.language import Language
from app.models import RetrievedDocument, ReviewDecision, WorkflowOptions
from app.prompts.templates import ReflectionAgentPrompts
from app.prompts.localized import get_prompts


class ReflectionAgent:
    """
    Agent for reviewing and reflecting on search results.
    Determines quality of results and whether to continue or finalize.
    """
    
    def __init__(self, settings: Settings, logger: Logger, workflow_options: WorkflowOptions):
        """Initialize the reflection agent using MAF ChatAgent."""
        self.settings = settings
        self.logger = logger
        self.workflow_options = workflow_options
        self.tracer = trace.get_tracer("ReflectionAgent")
        
        # Initialize MAF ChatAgent
        credential = DefaultAzureCredential() if settings.use_managed_identity else None
        
        chat_client = AzureOpenAIChatClient(
            credential=credential,
            endpoint=settings.azure_openai.endpoint,
            api_version=settings.azure_openai.api_version,
            deployment_name=settings.azure_openai.deployment_name
        )

        # One ChatAgent per supported language for review/decision in the
        # corpus language (the JSON schema itself is language-neutral).
        from typing import Dict as _Dict
        self._chat_client = chat_client
        self._agents: _Dict[Language, ChatAgent] = {
            Language.ENGLISH: ChatAgent(
                chat_client=chat_client,
                name="ReflectionAgent",
                instructions=get_prompts(Language.ENGLISH).reflection_system,
            ),
            Language.FRENCH_CANADIAN: ChatAgent(
                chat_client=chat_client,
                name="ReflectionAgent_FR",
                instructions=get_prompts(Language.FRENCH_CANADIAN).reflection_system,
            ),
        }
        self.agent = self._agents[Language.ENGLISH]
        
        self.logger.info(f"ReflectionAgent initialized with MAF ChatAgent: {settings.azure_openai.deployment_name}")
    
    async def review_search_results(
        self,
        user_query: str,
        current_results: List[RetrievedDocument],
        vetted_results: List[RetrievedDocument],
        search_history: List[Dict[str, Any]],
        max_attempts: int,
        current_attempt: int,
        language: str = "en",
    ) -> Tuple[ReviewDecision, List[RetrievedDocument], List[RetrievedDocument], str]:
        """Review search results and determine next action."""
        lang = Language.from_value(language)
        self.logger.info(
            f"Reviewing {len(current_results)} search results "
            f"(attempt {current_attempt}/{max_attempts}, language={lang.short_code})"
        )
        
        try:
            # Format results for review
            current_formatted = self._format_results(current_results)
            vetted_formatted = self._format_results(vetted_results)
            history_formatted = self._format_search_history(search_history)
            
            # Build review prompt
            review_prompt = ReflectionAgentPrompts.build_review_prompt(
                user_query=user_query,
                current_results_formatted=current_formatted,
                vetted_results_formatted=vetted_formatted,
                vetted_results_count=len(vetted_results),
                search_history_formatted=history_formatted,
                current_results_count=len(current_results),
                current_attempt=current_attempt,
                max_attempts=max_attempts
            )
            
            # Get review decision from LLM using MAF ChatAgent with structured output
            #message = ChatMessage(role=Role.USER, text=review_prompt)
            
            message = ChatMessage(role=Role.USER, text=review_prompt)
            agent = self._agents.get(lang, self._agents[Language.ENGLISH])
            response = await agent.run(
                messages=[message],
                options={
                    "response_format": ReviewDecision          
                },
            )

            decision: ReviewDecision | None = response.value

            # If structured parsing failed, try to recover from raw text.
            if decision is None:
                self.logger.warning(
                    "Structured output parsing returned None; attempting JSON fallback parse. "
                    f"Raw response text starts with: {response.text[:200]!r}"
                )
                try:
                    decision = ReviewDecision.model_validate_json(response.text)
                except Exception as parse_exc:
                    raise ValueError(f"Failed to parse ReviewDecision from response text: {parse_exc}")

            # Store original LLM decision before any override
            llm_original_decision = decision.decision
            
            # Filter valid indices to prevent IndexError
            valid_indices = [idx for idx in decision.valid_results if 0 <= idx < len(current_results)]
            invalid_indices = [idx for idx in decision.invalid_results if 0 <= idx < len(current_results)]
            
            # Apply smart retry logic
            final_decision = self._apply_smart_retry_logic(
                decision=decision.decision,
                valid_count=len(valid_indices),
                total_count=len(current_results),
                current_attempt=current_attempt,
                max_attempts=max_attempts
            )
            
            # Update decision if overridden
            if final_decision != llm_original_decision:
                decision.decision = final_decision
            
            # Separate valid and invalid results
            new_vetted = [current_results[idx] for idx in valid_indices]
            discarded = [current_results[idx] for idx in invalid_indices]
            
            self.logger.info(
                f"Review complete: {len(new_vetted)}/{len(current_results)} valid "
                f"({len(new_vetted)/len(current_results)*100:.1f}%), "
                f"LLM decision: {llm_original_decision}, Final decision: {decision.decision}"
            )
            
            return decision, new_vetted, discarded, llm_original_decision
            
        except Exception as e:
            self.logger.error(f"Review failed: {e}")
            # Fallback: accept all results and finalize
            fallback_decision = ReviewDecision(
                thought_process=f"Review failed: {str(e)}. Accepting all results.",
                valid_results=list(range(len(current_results))),
                invalid_results=[],
                decision="finalize"
            )
            return fallback_decision, current_results, [], "finalize"
    
    def _format_results(self, results: List[RetrievedDocument]) -> str:
        """Format results for display in prompts."""
        if not results:
            return "No results available."
        
        parts = []
        for i, doc in enumerate(results):
            parts.append(f"\nResult #{i}")
            parts.append("=" * 80)
            parts.append(f"Content ID: {doc.content_id}")
            parts.append(f"Document ID: {doc.document_id}")
            parts.append(f"Title: {doc.title}")
            parts.append(f"Score: {doc.score:.4f}")
            if doc.reranker_score is not None:
                parts.append(f"Reranker Score: {doc.reranker_score:.4f}")
            parts.append("\n--- Content ---")
            parts.append(doc.content)
            parts.append("--- End Content ---")
            parts.append("-" * 80)
        
        return "\n".join(parts)
    
    def _format_search_history(self, search_history: List[Dict[str, Any]]) -> str:
        """Format search history for context."""
        if not search_history:
            return "No previous search attempts."
        
        parts = ["\n=== Search History ==="]
        for i, entry in enumerate(search_history, 1):
            parts.append(f"\n<Attempt {i}>")
            parts.append(f"Query: {entry.get('query', '')}")
            parts.append(f"Review: {entry.get('review', '')}")
            parts.append("</Attempt>")
        
        return "\n".join(parts)
    
    def _apply_smart_retry_logic(
        self,
        decision: Literal["retry", "finalize"],
        valid_count: int,
        total_count: int,
        current_attempt: int,
        max_attempts: int
    ) -> Literal["retry", "finalize"]:
        """Apply intelligent retry logic to override LLM decision if needed.

        Overrides:
        - ``finalize`` → ``retry``: when the hit-rate is high enough that more
          relevant content likely exists in the index.        
        """
        if total_count == 0:
            return "retry" if current_attempt < max_attempts else "finalize"

        valid_percentage = valid_count / total_count

        high = self.workflow_options.reflection_high_validity_threshold
        moderate = self.workflow_options.reflection_moderate_validity_threshold
        moderate_min = self.workflow_options.reflection_moderate_validity_min_count        
          
        valid_percentage = valid_count / total_count
        
        # Override finalize if we're finding lots of valid content (suggests more exists)
        if decision == "finalize" and current_attempt < max_attempts:
            if valid_percentage >= high:
                self.logger.info(f"Overriding 'finalize': High validity rate ({valid_percentage:.1%}) suggests more content available")
                return "retry"
            elif valid_percentage >= moderate and valid_count >= moderate_min:
                self.logger.info(f"Overriding 'finalize': Good validity ({valid_percentage:.1%}) with {valid_count} results suggests comprehensive search needed")
                return "retry"
        
        return decision
