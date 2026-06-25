"""Domain models for chat, RAG workflows, and conversation history.

This module contains all Pydantic models related to chat functionality,
including agentic RAG workflow state, document retrieval, and conversation storage.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source document citation with metadata.
    
    Used throughout the application for tracking source attribution
    in generated responses and workflow execution.
    
    Attributes:
        document_id: Unique identifier for the source document (e.g., "doc-123-abc")
        content_id: Unique identifier for the content chunk (e.g., "content-123-abc")
        content: The actual text content of the citation.
        document_title: Title of the source document.
        page_number: Page number where the content appears.
    """
    document_id: str = Field(..., description="Unique identifier for the source document")
    content_id: str = Field(..., description="Unique identifier for the content chunk")
    content: Optional[str] = Field(
        default=None,
        description="The actual text content of the citated chunk"
    )
    document_title: Optional[str] = Field(
        default=None,
        description="Title of the source document"
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Page number where the content appears"
    )
    blob_url: Optional[str] = Field(
        default=None,
        description="Resolved Azure Blob Storage URL for this citation"
    )


class RetrievedDocument(BaseModel):
    """Retrieved document with metadata."""
    document_id: str
    content_id: str
    title: str
    content: str
    source: str
    page_number: Optional[int] = None
    score: float
    reranker_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatHistoryItem(BaseModel):
    """Cosmos DB chat history item.
    
    Represents a single message in conversation history stored in Cosmos DB.
    Uses Hierarchical Partition Key (HPK) with user_id and session_id for efficient querying.
    
    Attributes:
        id: Unique item ID in format {user_id}_{session_id}_{message_id}
        user_id: User identifier (partition key level 1)
        session_id: Session identifier (partition key level 2)
        timestamp: Message timestamp in UTC
        serialized_message: JSON-serialized message data
        message_text: Plain text of the message for quick access
        message_id: Unique message identifier
        role: Message role (user, assistant, system)
    """
    id: str = Field(..., description="Unique item ID")
    user_id: str = Field(..., description="User identifier (partition key level 1)")
    session_id: str = Field(..., description="Session identifier (partition key level 2)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Message timestamp"
    )
    serialized_message: str = Field(..., description="JSON-serialized message")
    message_text: Optional[str] = Field(default=None, description="Message text")
    message_id: str = Field(..., description="Unique message ID")
    role: str = Field(..., description="Message role")


class QueryType(str, Enum):
    """Query classification types."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    CONVERSATIONAL = "conversational"


class RewrittenQuery(BaseModel):
    """Output of QueryRewriterAgent."""
    hypothetical_passage: str
    reasoning: str


class GeneratedAnswer(BaseModel):
    """Generated answer with citations."""
    answer_text: str
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReviewDecision(BaseModel):
    """LLM decision for reviewing search results.
    
    Used by reflection agent to classify search results as valid/invalid
    and decide whether to continue searching or finalize.
    """
    thought_process: str
    valid_results: List[int] # Indices of valid results
    invalid_results: List[int] # Indices of invalid results
    decision: Literal["retry", "finalize"]
    

class AgenticRAGState(BaseModel):
    """State for agentic RAG workflow.
    
    Tracks the complete state through search → reflection → answer generation cycles.
    The thought_process captures each step with detailed metadata for observability.
    """
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None
    language: str = Field(
        default="en",
        description="Content language for retrieval and generation ('en' or 'fr-ca')",
    )
    
    # Iteration control
    max_attempts: int = 3
    current_attempt: int = 0
    search_history: List[Dict[str, Any]] = Field(default_factory=list)
    previous_reviews: List[str] = Field(default_factory=list)  # Review text from LLM
    decisions: List[str] = Field(default_factory=list)  # Track decision per iteration
    
    # Results tracking
    current_results: List[RetrievedDocument] = Field(default_factory=list)
    vetted_results: List[RetrievedDocument] = Field(default_factory=list)
    discarded_results: List[RetrievedDocument] = Field(default_factory=list)
    processed_content_ids: set = Field(default_factory=set)  # Stores content_ids to track unique chunks already processed
    
    # Tracks whether filters were bypassed and an unfiltered search was performed
    searched_without_filters: bool = False

    answer_retry_attempted: bool = False
    
    # Decision - drives workflow routing
    decision: Literal["search", "reflect", "finalize", "answer", "retry_no_filters"] = "search"
    
    # Final output
    answer: Optional[str] = None
    citations: Optional[List[Citation]] = None
    
    # Thought process - detailed step-by-step execution log
    # Each entry has {"step": str, "details": dict, "attempt": int (optional)}
    # Example steps: "retrieve", "review", "answer_generation"
    thought_process: List[Dict[str, Any]] = Field(default_factory=list)
    
