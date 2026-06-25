"""Shared FastAPI dependencies for dependency injection.

This module provides reusable dependency functions that can be injected
into route handlers using FastAPI's Depends() mechanism.
"""

from fastapi import Depends

from app.core.container import Container
from app.core.logger import Logger
from app.core.settings import Settings
from app.services.chat_service import IChatService
from app.services.conversation_service import IConversationService
from app.ingestion.indexer_service import IIndexerService
from app.ingestion.search_pipeline_orchestrator import ISearchPipelineOrchestrator


def get_container() -> Container:
    """Get the DI container for dependency injection."""
    from app.api.main import container
    return container


def get_logger() -> Logger:
    """Get the logger for dependency injection."""
    from app.api.main import logger
    return logger


def get_settings(container: Container = Depends(get_container)) -> Settings:
    """Get the application settings from the DI container."""
    return container.config()


def get_chat_service(container: Container = Depends(get_container)) -> IChatService:
    """Get the chat service from the DI container."""
    return container.chat_service()


def get_conversation_service(container: Container = Depends(get_container)) -> IConversationService:
    """Get the conversation service from the DI container."""
    return container.conversation_service()


def get_indexer_service(container: Container = Depends(get_container)) -> IIndexerService:
    """Get the indexer service from the DI container."""
    return container.indexer_service()


def get_search_pipeline_orchestrator(container: Container = Depends(get_container)) -> ISearchPipelineOrchestrator:
    """Get the search pipeline orchestrator from the DI container."""
    return container.search_pipeline_orchestrator()
