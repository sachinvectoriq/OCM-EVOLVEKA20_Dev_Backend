"""Knowledge Assistant Agentic RAG Application Package.

This package contains the complete backend application with:
- FastAPI REST API for conversational AI with agentic RAG
- Document ingestion pipeline orchestration via HTTP endpoints
- Shared services, models, and core infrastructure

Package Structure:
    api/         - FastAPI application (routes, schemas, main)
    agents/      - AI agent implementations
    core/        - Core infrastructure (settings, logging, DI container, exceptions)
    services/    - Business logic services (agents, evaluation)
    ingestion/   - Document ingestion pipeline (indexer, skillset, data source, orchestrator)
    models/      - Pydantic configuration models (search, blob, AI services, OpenAI)
    utils/       - Shared utilities and helpers

Architecture:
    - FastAPI with automatic OpenAPI documentation
    - Dependency injection with dependency-injector
    - Pydantic Settings for configuration from environment variables
    - Azure AI Search for vector and hybrid search with multimodal skillsets
    - Azure OpenAI for embeddings and chat completion
    - Azure Blob Storage for document and image storage
    - Async/await patterns for I/O operations
"""

__all__ = ["api", "agents", "core", "services", "ingestion", "models", "utils"]
