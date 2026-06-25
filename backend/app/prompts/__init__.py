"""Prompt templates for agents and workflow stages."""

from app.prompts.templates import (
    RAG_ASSISTANT_SYSTEM_PROMPT,
    ReflectionAgentPrompts,
    QueryRewriterPrompts,
    AnswerGeneratorPrompts,
    IngestionPrompts,
    FOLLOWUP_QUESTIONS_PROMPT,
)
from app.prompts.localized import (
    LocalizedPrompts,
    IngestionPromptsBundle,
    IngestionPromptsFR,
    get_prompts,
    get_ingestion_prompts,
)

__all__ = [
    "RAG_ASSISTANT_SYSTEM_PROMPT",
    "ReflectionAgentPrompts",
    "QueryRewriterPrompts",
    "AnswerGeneratorPrompts",
    "IngestionPrompts",
    "IngestionPromptsFR",
    "FOLLOWUP_QUESTIONS_PROMPT",
    "LocalizedPrompts",
    "IngestionPromptsBundle",
    "get_prompts",
    "get_ingestion_prompts",
]
