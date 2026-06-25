"""Agentic RAG workflows for query processing and answer generation.

This module exports the AgenticRAGWorkflow class for orchestrating multi-agent
RAG pipelines using the Microsoft Agent Framework with dependency injection.
"""

from app.workflows.agentic_rag_workflow import AgenticRAGWorkflow

__all__ = [
    "AgenticRAGWorkflow",
]
