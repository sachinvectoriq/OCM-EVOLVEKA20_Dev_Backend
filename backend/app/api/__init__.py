"""FastAPI application module for Knowledge Assistant.

This package contains the FastAPI application implementation including:
- main: FastAPI app factory and configuration
- routes: API endpoint handlers (chat, health, pipeline)
- schemas: Pydantic models for request/response validation
- dependencies: Dependency injection utilities

Note: Use explicit imports (e.g., from app.api.main import create_app)
to avoid circular import issues with the DI container.
"""

# Avoid eager imports that can cause circular dependencies
# Import create_app explicitly when needed: from app.api.main import create_app

__all__ = ["main", "routes", "schemas"]
