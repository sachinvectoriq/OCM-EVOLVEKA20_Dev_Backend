


"""Chat service for RAG-based conversational AI.

This module orchestrates the Retrieval-Augmented Generation (RAG) flow
by combining search results with LLM generation for grounded responses.

Purpose:
    - Retrieve relevant context from Azure AI Search
    - Build prompts with retrieved context
    - Generate responses using Azure OpenAI
    - Return responses with source citations

Usage:
    chat_service = ChatService(search_service, openai_options)
    response = await chat_service.chat_async(
        message="What are Azure best practices?",
        conversation_history=[...],
        filters=SearchFilters(opco_values=["TEKsystems"]),
        top_k=5
    )
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, cast, Iterable
import uuid

import tiktoken
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from agent_framework import Workflow, WorkflowOutputEvent, AgentRunUpdateEvent
from app.workflows.agentic_rag_workflow import AgenticRAGWorkflow
from agent_framework import ChatMessage as MAFChatMessage, Role
import json
import base64
import re
from urllib.parse import unquote

from app.core.logger import Logger
from app.api.schemas.chat import (
    ChatMessage,
    ChatResponse,
    Citation,
    SearchFilters,
    QueryResponse,
    ConversationMessage,
)
from app.models import AzureOpenAIOptions, RetrievedDocument, AgenticRAGState
from app.models import WorkflowOptions
from app.prompts.templates import RAG_ASSISTANT_SYSTEM_PROMPT
from app.services.search_service import ISearchService
from app.utils.blob_sas import sign_blob_url
from app.utils.citation_tracker import CitationTracker


# Token limit constants for different models
# These are conservative estimates leaving room for response generation
MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000,  # 128k context window
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
}

# Default fallback for unknown models (same as gpt-4o)
DEFAULT_MODEL_TOKEN_LIMIT = 128000

# Reserve tokens for response generation and safety margin
RESERVED_TOKENS_FOR_RESPONSE = 2000
SAFETY_MARGIN_TOKENS = 500

# Maximum tokens to use for context from search results
# This leaves room for system prompt, conversation history, and response
DEFAULT_MAX_CONTEXT_TOKENS = 25000  # ~100k tokens for gpt-4o with 128k window

# Approximate character-to-token ratio for English text
# This is a rough estimate; actual ratios vary by content type and language
APPROX_CHARS_PER_TOKEN = 4

def decode_blob_url(encoded_string):
    """
    Ultra-robust blob URL decoder.

    Handles:
    - _text_sections_X
    - _normalized_images_X
    - corrupted padding
    - URL-safe base64 (- and _)
    - random trailing garbage
    - malformed pdf suffixes
    - strange future variations

    Always returns the clean Azure Blob URL.
    """

    # ---------------------------
    # Step 1: Split prefix safely
    # ---------------------------
    try:
        _, payload = encoded_string.split('_', 1)
    except ValueError:
        raise ValueError("Invalid format. Expected: prefix_base64payload")

    # -------------------------------------------------------
    # Step 2: Extract ONLY continuous valid Base64 characters
    # -------------------------------------------------------
    # Accept both standard and URL-safe Base64
    b64_match = re.match(r'^[A-Za-z0-9+/_=-]+', payload)
    if not b64_match:
        raise ValueError("No valid Base64 segment found")

    b64_payload = b64_match.group(0)

    # ---------------------------------------
    # Step 3: Convert URL-safe → standard
    # ---------------------------------------
    b64_payload = b64_payload.replace('-', '+').replace('_', '/')

    # ---------------------------------------
    # Step 4: Strip known metadata suffixes
    # ---------------------------------------
    # Remove common trailing patterns if embedded
    b64_payload = re.sub(r'(text_sections_\d+.*)$', '', b64_payload)
    b64_payload = re.sub(r'(normalized_images_\d+.*)$', '', b64_payload)

    # ---------------------------------------
    # Step 5: Fix padding safely
    # ---------------------------------------
    b64_payload = re.sub(r'[^A-Za-z0-9+/=]', '', b64_payload)
    b64_payload += '=' * (-len(b64_payload) % 4)

    # ---------------------------------------
    # Step 6: Decode safely
    # ---------------------------------------
    try:
        raw_bytes = base64.b64decode(b64_payload, validate=False)
    except Exception:
        # If still broken, truncate until decodable
        for i in range(len(b64_payload), 0, -1):
            try:
                raw_bytes = base64.b64decode(b64_payload[:i], validate=False)
                break
            except Exception:
                continue
        else:
            raise ValueError("Unable to decode Base64 payload")

    raw_url = raw_bytes.decode("utf-8", errors="ignore")

    # ---------------------------------------
    # Step 7: Extract actual HTTPS blob URL
    # ---------------------------------------
    url_match = re.search(r'https?://[^\s<>"]+', raw_url)
    if not url_match:
        raise ValueError("No valid URL found in decoded data")

    clean_url = unquote(url_match.group(0))

    # ---------------------------------------
    # Step 8: Clean malformed PDF suffixes
    # ---------------------------------------
    pdf_match = re.search(r'\.pdf[^\s]*', clean_url, re.IGNORECASE)
    if pdf_match:
        clean_url = clean_url[:pdf_match.start() + 4]

    return clean_url


async def _generate_followups_via_llm(
    client: Any,
    model: str,
    system_msg: str,
    user_prompt: str,
    logger: Any,
) -> List[str]:
    """
    Robustly call the chat completion API to generate follow-up questions.

    Handles:
      - Models that require `max_completion_tokens` instead of `max_tokens`
        (e.g. o1 / gpt-5 family on Azure).
      - Models that don't support `response_format={"type":"json_object"}`.
      - Models that don't support custom `temperature`.
      - Logs the actual exception type + message and the raw model output
        so we can see WHY follow-ups fail instead of silently returning [].
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    # Try progressively simpler call variants. Each tuple is a kwargs override.
    # We start with the strictest (json_object + max_tokens) and fall back.
    attempts: List[Dict[str, Any]] = [
        # gpt-5 / o-series: max_completion_tokens, no custom temperature
        {"max_completion_tokens": 400, "response_format": {"type": "json_object"}},
        {"max_completion_tokens": 400},
        # Older / gpt-4o-style deployments
        {"max_tokens": 400, "temperature": 0.7, "response_format": {"type": "json_object"}},
        {"max_tokens": 400, "temperature": 0.7},
        {"max_tokens": 400},
    ]

    last_err: Optional[BaseException] = None
    raw_content: str = ""
    for kwargs in attempts:
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=cast(Iterable[ChatCompletionMessageParam], messages),
                **kwargs,
            )
            raw_content = (completion.choices[0].message.content or "") if completion.choices else ""
            parsed = _parse_followup_questions(raw_content)
            if parsed:
                return parsed
            # Empty parse — try next variant only if it's a plausibly different shape;
            # otherwise return empty list and log raw content for debugging.
            logger.warning(
                f"[Followups] Parsed 0 questions. variant={list(kwargs.keys())} "
                f"raw_preview={raw_content[:300]!r}"
            )
            # Don't keep retrying once we got a successful response — model just refused.
            return []
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(
                f"[Followups] LLM call failed (variant={list(kwargs.keys())}): "
                f"{type(e).__name__}: {e}"
            )
            continue

    if last_err is not None:
        logger.error(f"[Followups] All variants failed. last_error={type(last_err).__name__}: {last_err}")
    return []


def _parse_followup_questions(raw_content: str) -> List[str]:
    """Parse follow-up questions from raw model output, tolerating fenced JSON."""
    content = (raw_content or "").strip()
    if not content:
        return []

    try:
        payload = json.loads(content)
    except Exception:
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = content.find("{")
            end = content.rfind("}")
            candidate = content[start:end + 1] if start != -1 and end != -1 and end > start else None
        if not candidate:
            return []
        try:
            payload = json.loads(candidate)
        except Exception:
            return []

    questions = payload.get("follow_up_questions", []) if isinstance(payload, dict) else []
    if not isinstance(questions, list):
        return []
    return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:3]

class IChatService(ABC):
    """Interface for chat service operations."""

    @abstractmethod
    async def chat_async(
        self,
        message: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        filters: Optional[SearchFilters] = None,
        top_k: int = 5,
        language: str = "en",
    ) -> ChatResponse:
        """
        Process a chat message and generate a RAG-based response.

        Args:
            message: The user's question or message.
            conversation_history: Optional previous messages for context.
            filters: Optional search filters to narrow results.
            top_k: Number of search results to retrieve.

        Returns:
            ChatResponse with the AI-generated answer and citations.
        """
        pass

    @abstractmethod
    async def query_async(
        self,
        query: str,        
        session_id: str,
        user_id: Optional[str] = None,
        conversation_history: Optional[List[ConversationMessage]] = None,
        filters: Optional[Any] = None,
        language: str = "en",
    ) -> QueryResponse:
        """
        Execute agentic RAG query using workflow orchestration.

        Args:
            query: The user's question or query.
            user_id: Optional user identifier (should be pulled from JWT claim when auth is enabled).
            session_id: Session ID (must be valid UUID format) for conversation context.
            conversation_history: Optional list of previous messages.
            filters: Optional search filters (opco_values, persona_values).

        Returns:
            QueryResponse with answer, citations, and workflow metadata.
        """
        pass

    @abstractmethod
    async def list_conversations(
        self,
        user_id: str,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List all conversations for a user."""
        pass

    @abstractmethod
    async def get_conversation(
        self,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Get conversation metadata."""
        pass

    @abstractmethod
    async def delete_conversation(
        self,
        user_id: str,
        session_id: str,
    ) -> None:
        """Delete a conversation."""
        pass

    @abstractmethod
    async def clear_user_history(
        self,
        user_id: str,
    ) -> int:
        """Clear all conversation history for a user."""
        pass


class ChatService(IChatService):
    """
    Service for RAG-based conversational AI.

    Orchestrates the full RAG pipeline:
    1. Search for relevant documents using hybrid search
    2. Build a prompt with retrieved context
    3. Generate a response using Azure OpenAI
    4. Return the response with source citations

    Features:
        - Context-aware responses grounded in retrieved documents
        - Conversation history support for multi-turn interactions
        - Source citations for transparency and verification
        - Configurable search filters and result count
    """

    def __init__(
        self,
        search_service: ISearchService,
        openai_options: AzureOpenAIOptions,
        logger: Logger,
        workflow_options: WorkflowOptions,
        conversation_service=None,  # Optional ConversationService for conversation history
        workflow: Optional[AgenticRAGWorkflow] = None,  # Optional workflow for agentic RAG
        citation_tracker: Optional[CitationTracker] = None,  # Optional citation tracker
    ) -> None:
        """
        Initialize the ChatService.

        Args:
            search_service: Service for searching the knowledge base.
            openai_options: Configuration for Azure OpenAI chat completion.
            logger: Injected logging service.
            workflow_options: Workflow execution configuration.
            conversation_service: Optional Cosmos DB conversation service for managing history.
            workflow: Optional MAF workflow instance for agentic RAG orchestration.
            citation_tracker: Optional citation tracker for source attribution.
        """
        self._search_service: ISearchService = search_service
        self._openai_options: AzureOpenAIOptions = openai_options
        self.logger: Logger = logger
        self._workflow_options: WorkflowOptions = workflow_options
        self._conversation_service = conversation_service
        self._workflow_builder = workflow
        self._workflow: Optional[Workflow] = None
        
        if workflow:
            # Support both AgenticRAGWorkflow class instances and direct Workflow instances (for tests)
            if hasattr(workflow, 'build_workflow'):
                self._workflow = workflow.build_workflow()
            else:
                # Assume it's a Workflow instance directly (e.g., DummyWorkflow in tests)
                self._workflow = workflow  # type: ignore
                
        self._citation_tracker = citation_tracker or CitationTracker(logger)
        self._openai_client: Optional[AsyncAzureOpenAI] = None
        self._tokenizer: Optional[tiktoken.Encoding] = None

    async def _get_openai_client(self) -> AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client for chat completion (lazy initialization)."""
        if self._openai_client is None:
            # Use effective properties that handle fallback logic
            api_key = self._openai_options.effective_chat_completion_key
            endpoint = self._openai_options.resource_uri  # Base endpoint for client

            if api_key:
                self._openai_client = AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version="2024-06-01",
                )
            else:
                # Use managed identity
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                self._openai_client = AsyncAzureOpenAI(
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version="2024-06-01",
                )
        return self._openai_client

    def _get_tokenizer(self) -> tiktoken.Encoding:
        """Get or create the tokenizer for the current model."""
        if self._tokenizer is None:
            try:
                # Try to get encoding for the specific model
                model_name = self._openai_options.chat_completion_model.lower()
                # Map common model names to their encodings
                if "gpt-4" in model_name or "gpt-3.5" in model_name:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
            except (KeyError, Exception) as e:
                # Fallback to cl100k_base encoding (used by gpt-4, gpt-3.5-turbo, etc.)
                self.logger.warning(
                    f"Unable to get encoding for model {self._openai_options.chat_completion_model}, "
                    f"using cl100k_base encoding as fallback: {e}"
                )
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens in the text.
        """
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))

    def _estimate_message_tokens(
        self,
        message: str,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> int:
        """
        Estimate the number of tokens used by the message and conversation history.

        Args:
            message: The current user message.
            conversation_history: Optional previous messages.

        Returns:
            Estimated token count for messages (excluding context).
        """
        total_tokens = self._count_tokens(message)
        
        if conversation_history:
            for msg in conversation_history:
                total_tokens += self._count_tokens(msg.content)
                # Add overhead for role and formatting (~4 tokens per message)
                total_tokens += 4
        
        # Add overhead for current message formatting
        total_tokens += 4
        
        return total_tokens

    def _get_max_context_tokens(self) -> int:
        """
        Get the maximum number of tokens allowed for context based on the model.

        Returns:
            Maximum context tokens available for search results.
        """
        model_name = self._openai_options.chat_completion_model
        
        # Get model limit, default to standard limit if model not found
        model_limit = MODEL_TOKEN_LIMITS.get(model_name, DEFAULT_MODEL_TOKEN_LIMIT)
        
        # Calculate available tokens for context
        # Total limit - system prompt overhead - reserved for response - safety margin
        system_prompt_tokens = self._count_tokens(RAG_ASSISTANT_SYSTEM_PROMPT.format(context=""))
        max_context = model_limit - system_prompt_tokens - RESERVED_TOKENS_FOR_RESPONSE - SAFETY_MARGIN_TOKENS
        
        # Cap at default max to avoid excessive context
        return min(max_context, DEFAULT_MAX_CONTEXT_TOKENS)
        
  



    def _build_context_from_results(
        self,
        results: List[RetrievedDocument],
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build a formatted context string from search results with token limit.

        Iteratively adds search results to context until token limit is reached.
        Results are added in order of relevance (as provided in the list).

        Args:
            results: List of search results to include in context.
            max_tokens: Maximum number of tokens for the context. If None, uses model-based limit.

        Returns:
            Formatted context string for the prompt, truncated to fit token limit.
        """
        if not results:
            return "No relevant documents found."

        # Get max tokens if not provided
        if max_tokens is None:
            max_tokens = self._get_max_context_tokens()

        context_parts = []
        total_tokens = 0
        included_results = 0

        for i, result in enumerate(results, 1):
            title = result.title or "Unknown Document"
            page_info = f" (Page {result.metadata.get('page_number')})" if result.metadata.get('page_number') else ""
            
            # Format this result
            result_text = f"[Source {i}: {title}{page_info}]\n{result.content}\n"
            
            # Count tokens for this result including separator
            separator = "\n---\n" if context_parts else ""
            result_tokens = self._count_tokens(separator + result_text)
            
            # Check if adding this result would exceed the limit
            if total_tokens + result_tokens > max_tokens:
                self.logger.warning(
                    f"Context truncated at {included_results}/{len(results)} documents "
                    f"({total_tokens} tokens, limit: {max_tokens})"
                )
                break
            
            context_parts.append(result_text)
            total_tokens += result_tokens
            included_results += 1

        if included_results == 0:
            # Even the first result is too large, include a truncated version
            self.logger.warning("First search result exceeds token limit, truncating content")
            result = results[0]
            title = result.title or "Unknown Document"
            page_number = result.metadata.get('page_number')
            page_info = f" (Page {page_number})" if page_number else ""
            
            # Truncate content to fit within token limit
            header = f"[Source 1: {title}{page_info}]\n"
            header_tokens = self._count_tokens(header)
            available_tokens = max_tokens - header_tokens - 50  # Reserve some tokens for ellipsis
            
            # Start with character-based approximation for initial truncation
            approx_chars = available_tokens * APPROX_CHARS_PER_TOKEN
            truncated_content = result.content[:approx_chars]
            
            # Verify we're under the token limit
            truncated_text = header + truncated_content + "..."
            actual_tokens = self._count_tokens(truncated_text)
            
            # If still over limit, use binary search approach for efficiency
            if actual_tokens > max_tokens:
                left, right = 0, len(truncated_content)
                while left < right:
                    mid = (left + right + 1) // 2
                    test_content = result.content[:mid]
                    test_text = header + test_content + "..."
                    if self._count_tokens(test_text) <= max_tokens:
                        left = mid
                    else:
                        right = mid - 1
                truncated_content = result.content[:left]
            
            return header + truncated_content + "..."

        self.logger.info(
            f"Built context from {included_results}/{len(results)} documents "
            f"({total_tokens} tokens, limit: {max_tokens})"
        )
        
        return "\n---\n".join(context_parts)

    def _build_messages(
        self,
        message: str,
        context: str,
        conversation_history: Optional[List[ChatMessage]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the messages array for the OpenAI API call.

        Args:
            message: The current user message.
            context: The retrieved context to include in system prompt.
            conversation_history: Optional previous messages.

        Returns:
            List of message dictionaries for the API.
        """
        messages: List[Dict[str, Any]] = []

        # System message with context
        system_content = RAG_ASSISTANT_SYSTEM_PROMPT.format(context=context)
        messages.append({"role": "system", "content": system_content})

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        # Add current user message
        messages.append({"role": "user", "content": message})

        return messages

    async def chat_async(
        self,
        message: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        filters: Optional[SearchFilters] = None,
        top_k: int = 5,
        language: str = "en",
    ) -> ChatResponse:
        """
        Process a chat message and generate a RAG-based response.

        Pipeline:
        1. Search for relevant documents using hybrid search
        2. Calculate available tokens for context (accounting for conversation history)
        3. Build context from search results (truncated to fit token limits)
        4. Construct prompt with system instructions and context
        5. Generate response using Azure OpenAI
        6. Return response with source citations

        Token Management:
        - Automatically calculates available tokens based on model limits
        - Accounts for system prompt, conversation history, and response generation
        - Truncates context from search results if needed to stay within limits
        - Ensures at least minimum context is included even with long conversations

        Args:
            message: The user's question or message.
            conversation_history: Optional previous messages for context.
            filters: Optional search filters to narrow results.
            top_k: Number of search results to retrieve (default: 5).

        Returns:
            ChatResponse with the AI-generated answer and citations.

        Raises:
            Exception: If search or generation fails.
        """
        self.logger.info(f"Processing chat request: {message[:50]}...")

        # Step 1: Retrieve relevant documents
        # Build filters dictionary for new search signature
        search_filters = {}
        if filters:
            if filters.opco_values:
                search_filters["opco_values"] = filters.opco_values
            if filters.persona_values:
                search_filters["persona_values"] = filters.persona_values

        search_results = await self._search_service.search_async(
            query=message,
            top_k=top_k,
            search_mode="hybrid",
            filters=search_filters if search_filters else None,
            use_semantic_ranking=True,
            deduplicate=True
        )

        self.logger.info(f"Retrieved {len(search_results)} documents for context")

        # Step 2: Calculate available tokens for context
        # Account for conversation history and current message
        message_tokens = self._estimate_message_tokens(message, conversation_history)
        max_context_tokens = self._get_max_context_tokens()
        
        # Adjust context limit based on conversation history
        available_context_tokens = max(
            max_context_tokens - message_tokens,
            10000  # Minimum context tokens to ensure some context is included
        )
        
        self.logger.info(
            f"Token budget: message={message_tokens}, "
            f"context={available_context_tokens}, "
            f"max={max_context_tokens}"
        )

        # Step 3: Build context from results with token limit
        context = self._build_context_from_results(search_results, available_context_tokens)

        # Step 4: Build messages for OpenAI
        messages = self._build_messages(message, context, conversation_history)

        # Step 5: Generate response
        client = await self._get_openai_client()
        
        completion = await client.chat.completions.create(
            model=self._openai_options.chat_completion_model,
            messages=cast(Iterable[ChatCompletionMessageParam], messages),
            temperature=0.7,
            max_tokens=700,
        )

        response_text = completion.choices[0].message.content or ""
        self.logger.info(f"Generated response: {len(response_text)} characters")


        # Step 6: Create citations from search results
        citations = self._citation_tracker.create_citations(search_results)

        # Step 7: Generate follow-up questions using a new LLM call
        from app.prompts.localized import get_prompts as _get_prompts_for_lang
        from app.core.language import Language as _Lang
        _lang = _Lang.from_value(language)
        followup_prompt = _get_prompts_for_lang(language).followup_questions_template.format(
            answer=response_text, context=context
        )
        _system_msg = (
            "You are a helpful assistant. Always reply with valid JSON only."
            if _lang is _Lang.ENGLISH
            else "Tu es un assistant francophone (français canadien). Réponds UNIQUEMENT avec un JSON valide."
        )
        follow_up_questions = await _generate_followups_via_llm(
            client=client,
            model=self._openai_options.chat_completion_model,
            system_msg=_system_msg,
            user_prompt=followup_prompt,
            logger=self.logger,
        )

        return ChatResponse(
            message=response_text,
            citations=citations,
            follow_up_questions=follow_up_questions,
        )

    async def query_async(
        self,
        query: str,
        session_id: str,
        user_id: Optional[str] = None,        
        conversation_history: Optional[List[ConversationMessage]] = None,
        filters: Optional[Any] = None,
        language: str = "en",
    ) -> QueryResponse:
        """
        Execute agentic RAG query using MAF workflow orchestration.

        This method orchestrates the full agentic RAG pipeline:
        1. Load conversation history from Cosmos DB (if available)
        2. Execute the MAF workflow with agents for planning, retrieval, reflection, and generation
        3. Save conversation to Cosmos DB (if available)
        4. Return structured response with citations and metadata

        Args:
            query: The user's question or query.
            user_id: Optional user identifier (should be pulled from JWT claim when auth is enabled).
            session_id: Session ID (must be valid UUID format) for conversation context.
            conversation_history: Optional list of previous messages.
            filters: Optional search filters (opco_values, persona_values).

        Returns:
            QueryResponse with answer, citations, and workflow metadata.

        Raises:
            ValueError: If workflow is not initialized, session_id is invalid UUID, or session doesn't exist.
            Exception: If workflow execution fails.
        """
        if not self._workflow:
            raise ValueError("Workflow not initialized. Configure workflow in ChatService.")

        # Validate session_id is a valid UUID
        try:
            uuid.UUID(session_id)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid session_id format. Must be a valid UUID. Error: {e}")

        self.logger.info(f"Executing agentic RAG query: {query[:100]}...")

        # Load conversation history from Cosmos DB if available
        conv_history = None
        if self._conversation_service and user_id:
            try:
                conv_history = await self._conversation_service.get_conversation_history(
                    session_id=session_id,
                    user_id=user_id,
                    max_messages=self._workflow_options.conversation_history_window
                )
                self.logger.info(f"Loaded {len(conv_history)} messages from conversation history")
            except Exception as e:
                self.logger.warning(f"Failed to load conversation history: {e}")

        # Fallback to request conversation_history if provided
        if not conv_history and conversation_history:
            conv_history = [
                MAFChatMessage(
                    role=Role(msg.role.value),
                    text=msg.content
                )
                for msg in conversation_history
            ]

        # Convert conversation history to dict format for workflow state
        conversation_history_dict = []
        if conv_history:
            for msg in conv_history:
                conversation_history_dict.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.text
                })

        # Convert filters to dict format if provided
        filters_dict = None
        if filters:
            filters_dict = filters.model_dump() if hasattr(filters, 'model_dump') else filters

        # Create initial workflow state with configured max attempts
        initial_state = AgenticRAGState(
            query=query,
            user_id=user_id,
            session_id=session_id,
            conversation_history=conversation_history_dict,
            filters=filters_dict,
            max_attempts=self._workflow_options.max_retrieval_iterations,
            language=language,
        )

        # Execute workflow
        self.logger.info(f"[WORKFLOW START] Executing MAF workflow for query: '{query[:50]}...'")
        final_state = None

        async for event in self._workflow.run_stream(initial_state):
            # Log all events to debug
            self.logger.info(f"[WORKFLOW EVENT] {type(event).__name__}, origin: {getattr(event, 'origin', 'N/A')}")
            
            if isinstance(event, WorkflowOutputEvent):
                final_state = event.data
                self.logger.info("Workflow completed with output")
            elif isinstance(event, AgentRunUpdateEvent):
                self.logger.debug(f"Agent update: {event.executor_id}")

        if final_state is None:
            raise Exception("Workflow did not produce output")

        # Save conversation to Cosmos DB if available
        if self._conversation_service and user_id:
            try:
                # Save user message
                await self._conversation_service.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    content=query
                )

                # Save assistant response
                await self._conversation_service.add_message(
                    session_id=session_id,
                    user_id=user_id,
                    role="assistant",
                    content=final_state.answer or "Unable to generate answer",
                    metadata={
                        "citations": [c.model_dump() for c in (final_state.citations or [])],
                        "document_count": len(final_state.vetted_results or []),
                        "language": language,
                    }
                )
                self.logger.info(f"Saved conversation to Cosmos DB: {session_id}")
            except Exception as e:
                self.logger.error(f"Failed to save conversation: {e}")
        #Code added for decoding blob URLs in citations       
        decorated_citations = []
        source_citations = final_state.citations or []

        for c in source_citations:
            citation_dict = c.model_dump()

            try:
                raw_url = decode_blob_url(c.content_id)
                # Append a short-lived read-only SAS so the frontend / browser
                # can open the blob directly even when the container is
                # private. Falls back to the unsigned URL on any failure.
                citation_dict["blob_url"] = sign_blob_url(raw_url)
            except Exception as e:
                self.logger.warning(
                    f"[ChatService] Failed to decode blob URL for content_id={c.content_id}: {e}"
                )

            decorated_citations.append(citation_dict)


        # Generate follow-up questions using the answer and context chunks
        from app.prompts.localized import get_prompts as _get_prompts_for_lang
        # Use the answer and the concatenated vetted results as context
        context_chunks = "\n\n".join([str(chunk.content) for chunk in (final_state.vetted_results or [])])
        followup_prompt = _get_prompts_for_lang(language).followup_questions_template.format(
            answer=final_state.answer or "", context=context_chunks
        )
        followup_questions = []
        try:
            client = await self._get_openai_client()
            from app.core.language import Language as _Lang
            _lang = _Lang.from_value(language)
            _system_msg = (
                "You are a helpful assistant. Always reply with valid JSON only."
                if _lang is _Lang.ENGLISH
                else "Tu es un assistant francophone (français canadien). Réponds UNIQUEMENT avec un JSON valide."
            )
            if not (context_chunks or "").strip():
                self.logger.warning(
                    "[Followups] context_chunks is empty — vetted_results may be empty; "
                    "follow-ups will likely be empty per the strict prompt."
                )
            followup_questions = await _generate_followups_via_llm(
                client=client,
                model=self._openai_options.chat_completion_model,
                system_msg=_system_msg,
                user_prompt=followup_prompt,
                logger=self.logger,
            )
        except Exception as e:
            self.logger.warning(
                f"[Followups] Outer failure (agentic): {type(e).__name__}: {e}"
            )

        # Build response with rich workflow state data
        response = QueryResponse(
            answer=final_state.answer or "Unable to generate answer",
            citations=decorated_citations or [],
            follow_up_questions=followup_questions,
            document_count=len(final_state.vetted_results or []),
            session_id=session_id,
            thought_process=final_state.thought_process or [],
            search_history=final_state.search_history or [],
            decisions=final_state.decisions or [],
            attempts=final_state.current_attempt,
            language=language,
        )

        self.logger.info(f"Query completed: {len(response.answer)} chars, {len(response.citations)} citations")
        return response

    async def get_conversation(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific conversation with all messages.

        Args:
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            Dictionary with session_id and messages.

        Raises:
            ValueError: If chat store is not configured.
        """
        if not self._conversation_service:
            raise ValueError("Conversation history storage not available. Configure Cosmos DB.")

        messages = await self._conversation_service.get_conversation_history(
            session_id=session_id,
            user_id=user_id
        )
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "message_count": len(messages),
            "messages": [{
                "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                "text": msg.text,
                "id": msg.id
            } for msg in messages]
        }

    async def list_conversations(self, user_id: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List all conversation sessions for a user.

        Args:
            user_id: User identifier.
            max_results: Maximum number of sessions to return.

        Returns:
            List of conversation sessions with metadata.

        Raises:
            ValueError: If chat store is not configured.
        """
        if not self._conversation_service:
            raise ValueError("Conversation history storage not available. Configure Cosmos DB.")

        sessions = await self._conversation_service.list_user_sessions(user_id, max_results)
        return sessions or []

    async def delete_conversation(self, user_id: str, session_id: str):
        """
        Delete a specific conversation thread.

        Args:
            user_id: User identifier.
            session_id: Session identifier.

        Raises:
            ValueError: If chat store is not configured.
        """
        if not self._conversation_service:
            raise ValueError("Conversation history storage not available. Configure Cosmos DB.")

        await self._conversation_service.delete_session(session_id, user_id)

    async def clear_user_history(self, user_id: str) -> int:
        """
        Delete all conversation threads for a user.

        Args:
            user_id: User identifier.

        Returns:
            Number of conversations deleted.

        Raises:
            ValueError: If chat store is not configured.
        """
        if not self._conversation_service:
            raise ValueError("Conversation history storage not available. Configure Cosmos DB.")

        return await self._conversation_service.clear_user_history(user_id)
