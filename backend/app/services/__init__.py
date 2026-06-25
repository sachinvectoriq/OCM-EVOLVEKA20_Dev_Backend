"""Business logic services for Knowledge Assistant Agentic RAG.

This package contains service layer implementations that orchestrate
business logic and coordinate between core infrastructure and API layers.

Purpose:
    - Implement business logic separate from API routes and infrastructure
    - Coordinate between multiple Azure services
    - Provide testable, reusable service components
    - Manage stateful operations and workflows

Services:
    - SearchService: Hybrid search with vector + keyword + semantic ranking
    - ChatService: RAG-based conversational AI with citations
    - ConversationService: Cosmos DB conversation history management
"""

from app.services.search_service import (
    ISearchService,
    SearchService,
)
from app.services.chat_service import (
    IChatService,
    ChatService,
)
from app.services.conversation_service import (
    IConversationService,
    ConversationService,
)

__all__ = [
    "ISearchService",
    "SearchService",
    "IChatService",
    "ChatService",
    "IConversationService",
    "ConversationService",
]
