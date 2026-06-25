"""
Answer Generator Agent for synthesizing answers with citations.
Uses MAF ChatAgent for LLM interactions.
"""
import re
from typing import List, Dict, Optional
from azure.identity import DefaultAzureCredential

from agent_framework import ChatAgent, ChatMessage, Role
from agent_framework.azure import AzureOpenAIChatClient
from opentelemetry import trace

from app.core.settings import Settings
from app.core.logger import Logger
from app.core.language import Language
from app.models import (
    RetrievedDocument,
    GeneratedAnswer
)
from app.prompts.templates import AnswerGeneratorPrompts
from app.prompts.localized import get_prompts
from app.utils.citation_tracker import CitationTracker


class AnswerGenerator:
    """
    Agent for synthesizing answers with proper citation handling.
    Uses MAF ChatAgent for LLM interactions following framework patterns.
    
    Responsibilities:
    - Assemble context from retrieved documents
    - Generate comprehensive answers grounded in sources
    - Insert citations linking claims to sources
    - Handle cases with insufficient information
    - Control answer length based on query complexity
    """
    
    # Compiled regex patterns for efficiency
    _CITATION_PATTERN = re.compile(r"\{([^}]+)\}")
    _CITATION_NUMBER_PATTERN = re.compile(r'\[(\d+)\]')
    _CONSECUTIVE_CITATIONS_PATTERN = re.compile(r'(?:\[\d+\]){2,}')
    
    def __init__(self, settings: Settings, logger: Logger, citation_tracker: CitationTracker):
        """
        Initialize the answer generator agent using MAF ChatAgent.
        
        Args:
            settings: Application settings with Azure AI configuration
            logger: Injected logging service
            citation_tracker: Citation tracking utility for source attribution
        """
        self.settings = settings
        self.logger = logger
        self.citation_tracker = citation_tracker
        self.tracer = trace.get_tracer("AnswerGeneratorAgent")
        
        # Initialize MAF ChatAgent
        credential = DefaultAzureCredential() if settings.use_managed_identity else None
        
        chat_client = AzureOpenAIChatClient(
            credential=credential,
            endpoint=settings.azure_openai.endpoint,
            api_version=settings.azure_openai.api_version,
            deployment_name=settings.azure_openai.deployment_name
        )

        # Build one ChatAgent per supported language with the matching system
        # instructions so the model responds in the right tongue. The chat
        # client is shared.
        self._chat_client = chat_client
        self._agents: Dict[Language, ChatAgent] = {
            Language.ENGLISH: ChatAgent(
                chat_client=chat_client,
                name="AnswerGeneratorAgent",
                instructions=get_prompts(Language.ENGLISH).answer_generator_system,
            ),
            Language.FRENCH_CANADIAN: ChatAgent(
                chat_client=chat_client,
                name="AnswerGeneratorAgent_FR",
                instructions=get_prompts(Language.FRENCH_CANADIAN).answer_generator_system,
            ),
        }
        # Backwards-compatible attribute used by older call sites/tests.
        self.agent = self._agents[Language.ENGLISH]
        
        self.logger.info(f"AnswerGenerator initialized with MAF ChatAgent: {settings.azure_openai.deployment_name}")
    
    async def generate_answer(
        self,
        query: str,
        documents: List[RetrievedDocument],
        generated_answer_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        language: str = "en",
    ) -> GeneratedAnswer:
        """Generate answer from retrieved documents with citations.

        Args:
            language: Content language ("en" or "fr-ca"). Selects which
                ChatAgent (and therefore which system prompt and output
                language) to use.
        """
        try:
            lang = Language.from_value(language)
            # Log entry point to track duplicate calls
            self.logger.info(
                f"[GENERATE_ANSWER] Called with query: '{query[:50]}...', "
                f"{len(documents)} documents, language={lang.short_code}"
            )
            
            # Add custom attributes to current MAF span
            self.logger.add_span_attributes(
                operation="answer_generation",
                query_length=len(query),
                document_count=len(documents),
                agent_name="AnswerGeneratorAgent",
                language=lang.short_code,
            )
            
            # Handle case with no documents
            if not documents:
                return self._generate_fallback_answer(query, language=lang)
            
            # Generate answer using MAF ChatAgent
            response = await self._call_llm(
                generated_answer_prompt=generated_answer_prompt, 
                conversation_history=conversation_history,
                language=lang,
            )
            
            # Extract which documents were actually cited
            cited_docs = self._extract_cited_documents(response, documents)

            # Auto-retry once with stricter ref-N enforcement if no valid citations resolved
            if not cited_docs:
                self.logger.warning(
                    "[Citation] No valid citations resolved on first attempt — retrying with stricter ref-N enforcement"
                )
                stricter_prompt = (
                    generated_answer_prompt
                    + "\n\n=== STRICT RETRY INSTRUCTIONS ===\n"
                    + "Your previous response did not include any valid citations.\n"
                    + "You MUST now cite EVERY factual sentence using ONLY the short tokens "
                    + "shown as `Citation Token: ref-N` next to each Vetted Result.\n"
                    + "Format: place {ref-1}, {ref-2}, etc. at the END of each sentence, "
                    + "immediately after the period. Do NOT use the long Content ID. "
                    + "Do NOT invent ref-N values that were not provided. "
                    + "Do NOT place citations in the middle of a sentence.\n"
                    + "Re-answer the question now, applying these rules strictly."
                )
                response = await self._call_llm(
                    generated_answer_prompt=stricter_prompt,
                    conversation_history=conversation_history,
                    language=lang,
                )
                cited_docs = self._extract_cited_documents(response, documents)
   
            # Replace {Content Id} with [1], [2], [3] and sort consecutive citations
            final_answer = self._replace_content_with_indices(response, cited_docs, documents)
            
            # Create citations only for documents that were cited
            citations = self.citation_tracker.create_citations(cited_docs)
            
            generated_answer = GeneratedAnswer(
                answer_text=final_answer,
                citations=citations,
                metadata={
                    "document_count": len(documents),
                    "cited_count": len(cited_docs),
                    "language": lang.value,
                }
            )
                                    
            return generated_answer
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}", exc_info=True)
            return self._generate_fallback_answer(query, error=str(e), language=Language.from_value(language))
        
    
    def _generate_fallback_answer(
        self,
        query: str,
        error: Optional[str] = None,
        language: Language = Language.ENGLISH,
    ) -> GeneratedAnswer:
        """
        Generate fallback answer when documents unavailable.
        
        Args:
            query: User query
            error: Optional error message
        
        Returns:
            Fallback GeneratedAnswer
        """
        prompts = get_prompts(language)
        if error:
            answer_text = prompts.answer_generation_error_template.format(error=error)
        else:
            answer_text = prompts.no_documents_fallback

        self.logger.warning(
            f"Returning fallback answer ({language.short_code}) for query: {query[:100]}"
        )

        return GeneratedAnswer(
            answer_text=answer_text,
            citations=[],
            metadata={
                "fallback": True,
                "error": error,
                "language": language.value,
            }
        )
    
    async def _call_llm(
        self,
        generated_answer_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 300,
        language: Language = Language.ENGLISH,
    ) -> str:
        """Call LLM using MAF ChatAgent with generated answer prompt.
        
        The system prompt (closed-book rules, citation rules, etc.) is set at
        agent initialization. The generated_answer_prompt contains the user question,
        vetted results, and reflection analysis as a single user message.
        
        Args:
            generated_answer_prompt: Combined user question, vetted results, and reflection analysis
            conversation_history: Optional conversation history for additional context
            max_tokens: Maximum tokens for response
        
        Returns:
            LLM response text
        """
        self.logger.info(
            f"[CALL_LLM] Invoking MAF ChatAgent ({language.short_code}) with {max_tokens} max tokens"
        )
        
        # Build messages list starting with conversation history
        messages = []
        
        # Add conversation history if provided (citations already stripped at save time)
        if conversation_history:
            for msg in conversation_history:
                role = Role.USER if msg.get("role") == "user" else Role.ASSISTANT
                messages.append(ChatMessage(role=role, text=msg.get("content", "")))
        
        # Add generated answer prompt (includes user question + vetted results) as user message
        messages.append(ChatMessage(role=Role.USER, text=generated_answer_prompt))
        
        agent = self._agents.get(language, self._agents[Language.ENGLISH])
        # Run agent with constructed messages
        result = await agent.run(
            messages=messages,
            temperature=0.1,  # Low temperature for factual answers
            max_tokens=max_tokens
        )
        
        return result.messages[-1].text
    
    def _resolve_cited_id(
        self,
        cited_token: str,
        documents: List[RetrievedDocument],
    ) -> Optional[RetrievedDocument]:
        """
        Resolve a citation token from the LLM to a real RetrievedDocument.

        Resolution order:
          1. Short ref token (ref-1, ref_2, ref 3, REF1, even '{ref-1}') →
             documents[N-1], using only plain string operations (no regex).
          2. Exact content_id match.
          3. Tolerant substring / overlap match (rescues mangled content_ids).

        Args:
            cited_token: The raw token captured from inside `{...}`.
            documents: Available documents (1-based ref-N maps into this list).

        Returns:
            The matching RetrievedDocument, or None if nothing reasonable matches.
        """
        if not cited_token or not documents:
            return None

        # Normalize: strip whitespace and stray braces the LLM might include
        token = cited_token.strip().strip("{}").strip()
        if not token:
            return None

        # ---- 1) Try short ref-N token (no regex) ----
        lowered = token.lower()
        # Strip common separators between "ref" and the number
        if lowered.startswith("ref"):
            tail = lowered[3:]
            # Remove leading separators: '-', '_', ' ', ':', '#'
            while tail and tail[0] in ("-", "_", " ", ":", "#"):
                tail = tail[1:]
            # Trim trailing whitespace/punctuation
            tail = tail.strip().rstrip(".,;:)]}")
            if tail.isdigit():
                idx = int(tail)
                if 1 <= idx <= len(documents):
                    return documents[idx - 1]

        # ---- 2) Exact content_id match ----
        for doc in documents:
            if doc.content_id == token:
                return doc

        # ---- 3) Tolerant substring / overlap match ----
        # a) token is contained in a content_id, or content_id contained in token
        for doc in documents:
            cid = doc.content_id or ""
            if not cid:
                continue
            if token in cid or cid in token:
                return doc

        # b) longest common substring-ish overlap on a stable prefix/suffix.
        # Cheap heuristic: compare the first 12 and last 12 chars of each side.
        def _edges(s: str) -> tuple:
            s = s.strip()
            return (s[:12], s[-12:])

        t_head, t_tail = _edges(token)
        best: Optional[RetrievedDocument] = None
        best_score = 0
        for doc in documents:
            cid = doc.content_id or ""
            if not cid:
                continue
            c_head, c_tail = _edges(cid)
            score = 0
            if t_head and t_head == c_head:
                score += 2
            if t_tail and t_tail == c_tail:
                score += 2
            # partial edge match
            if t_head and (t_head in cid or c_head in token):
                score += 1
            if t_tail and (t_tail in cid or c_tail in token):
                score += 1
            if score > best_score:
                best_score = score
                best = doc

        if best is not None and best_score >= 2:
            return best

        return None

    def _extract_cited_documents(self, answer_text: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Extract which documents were actually cited in the answer.
        
        Parses {Content ID} patterns and matches to documents.
        
        Args:
            answer_text: Generated answer with {Content ID} citations
            documents: All documents
        
        Returns:
            List of documents that were cited
        """
        # Extract all {Content ID} patterns using pre-compiled regex
        cited_content_ids = self._CITATION_PATTERN.findall(answer_text)
        
        if not cited_content_ids:
            self.logger.warning("[Citation] No {Content ID} citations found in answer")
            return []
        
        # Create content ID to document mapping for fast lookup
        content_id_map = {doc.content_id: doc for doc in documents}
        
        # Get unique cited IDs preserving order
        unique_cited_ids = list(dict.fromkeys(cited_content_ids))
        
        # Log extracted content IDs (truncated for readability)
        self.logger.info(f"[Citation] Extracted {len(unique_cited_ids)} unique content IDs from answer:")
        for i, cid in enumerate(unique_cited_ids, 1):
            display_id = cid if len(cid) <= 50 else f"{cid[:25]}...{cid[-25:]}"
            self.logger.info(f"  [{i}] {display_id}")
        
        # Log available document content IDs
        self.logger.info(f"[Citation] Available documents: {len(documents)}")
        for i, doc in enumerate(documents, 1):
            display_id = doc.content_id if len(doc.content_id) <= 50 else f"{doc.content_id[:25]}...{doc.content_id[-25:]}"
            self.logger.info(f"  Doc[{i}] content_id: {display_id}")
        
        # Match cited tokens to documents (in order of first appearance) using the
        # tolerant resolver: short ref-N tokens, exact content_id, or overlap.
        cited_docs = []
        seen_doc_ids = set()
        unmatched_ids = set()  # Use set for O(1) lookup instead of list
        
        for cited_token in cited_content_ids:
            resolved = self._resolve_cited_id(cited_token, documents)
            if resolved is None:
                unmatched_ids.add(cited_token)
                continue
            if resolved.content_id in seen_doc_ids:
                continue
            cited_docs.append(resolved)
            seen_doc_ids.add(resolved.content_id)
        
        # Log unmatched content IDs
        if unmatched_ids:
            self.logger.warning(f"[Citation] {len(unmatched_ids)} content IDs could not be matched to documents:")
            for cid in unmatched_ids:
                display_id = cid if len(cid) <= 50 else f"{cid[:25]}...{cid[-25:]}"
                self.logger.warning(f"  UNMATCHED: {display_id}")
        
        self.logger.info(f"[Citation] Found {len(cited_docs)} cited documents from {len(cited_content_ids)} citations")
        return cited_docs
    
    def _replace_content_with_indices(
        self,
        answer_text: str,
        cited_docs: List[RetrievedDocument],
        documents: Optional[List[RetrievedDocument]] = None,
    ) -> str:
        """
        Replace {Content ID} / {ref-N} patterns with [n] citation indices and
        sort consecutive citations.

        Args:
            answer_text: Answer text with citation patterns.
            cited_docs: Cited documents in order (determines [n] numbering).
            documents: Full list of available documents, used for tolerant
                resolution of ref-N tokens. Falls back to ``cited_docs`` if not
                provided (for backward compatibility).

        Returns:
            Answer text with sorted [1], [2], [3] citations.
        """
        # Create content ID to index mapping (1-based)
        content_id_to_index = {doc.content_id: i + 1 for i, doc in enumerate(cited_docs)}
        resolution_pool = documents if documents else cited_docs

        self.logger.info(f"[Citation] Building index mapping for {len(cited_docs)} cited documents")

        # Track replacement stats
        replacements_made = 0
        replacements_failed = set()  # Use set for O(1) operations

        def replace_content_id(match):
            nonlocal replacements_made
            cited_token = match.group(1)

            resolved = self._resolve_cited_id(cited_token, resolution_pool)
            if resolved is not None and resolved.content_id in content_id_to_index:
                replacements_made += 1
                return f"[{content_id_to_index[resolved.content_id]}]"

            replacements_failed.add(cited_token)
            return ""
        
        # Replace all {Content ID} patterns with [n] (or remove if unmatched) using pre-compiled regex
        result = self._CITATION_PATTERN.sub(replace_content_id, answer_text)
        
        # Clean up extra whitespace from removed citations
        result = re.sub(r' {2,}', ' ', result)    # Multiple spaces → single space
        result = re.sub(r' \n', '\n', result)     # Space before newline → just newline
        result = re.sub(r'\n ', '\n', result)     # Space after newline → just newline
        
        # Log replacement summary
        self.logger.info(f"[Citation] Replacements: {replacements_made} successful, {len(replacements_failed)} failed")
        if replacements_failed:
            self.logger.warning(f"[Citation] Removed {len(replacements_failed)} unmatched citations (LLM hallucination - cited non-existent content IDs)")
            # Log only first few unmatched IDs to avoid log spam
            for cid in list(replacements_failed)[:5]:
                display_id = cid if len(cid) <= 50 else f"{cid[:25]}...{cid[-25:]}"
                self.logger.warning(f"  UNMATCHED: {display_id}")
            if len(replacements_failed) > 5:
                self.logger.warning(f"  ... and {len(replacements_failed) - 5} more")
        
        # Sort consecutive citations (e.g., [2][1] becomes [1][2])
        return self._sort_consecutive_citations(result)
    
    def _sort_consecutive_citations(self, answer_text: str) -> str:
        """
        Sort consecutive citation numbers to ensure proper ordering.
        
        Converts [2][1] to [1][2], [3][1][2] to [1][2][3], etc.
        
        Args:
            answer_text: Answer text with [n] citations
        
        Returns:
            Answer text with sorted consecutive citations
        """
        def sort_citations(match):
            # Extract all citation numbers from consecutive [n][n][n] pattern using pre-compiled regex
            citation_numbers = self._CITATION_NUMBER_PATTERN.findall(match.group(0))
            
            # Sort numerically and reconstruct as [1][2][3]
            return ''.join(f'[{n}]' for n in sorted(map(int, citation_numbers)))
        
        # Match consecutive citations using pre-compiled regex
        return self._CONSECUTIVE_CITATIONS_PATTERN.sub(sort_citations, answer_text)
