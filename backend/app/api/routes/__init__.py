"""API route modules for Knowledge Assistant.

This package contains all FastAPI route handlers organized by functionality:
- chat: Conversational AI endpoints with agentic RAG
- health: Application health and readiness probes
- pipeline: Document ingestion pipeline management endpoints

Each route module is registered with the main FastAPI app via APIRouter.
"""

from .chat import router as chat_router
from .health import router as health_router
from .pipeline import router as pipeline_router

__all__ = ["chat_router", "health_router", "pipeline_router"]

