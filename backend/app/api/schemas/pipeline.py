"""Pydantic schemas for pipeline endpoints."""

from typing import Optional

from pydantic import BaseModel, Field


class PipelineActionRequest(BaseModel):
    """Request model for pipeline actions.

    Attributes:
        reset: If true, resets the indexer before performing the action.
        language: Optional language code ("en" or "fr-ca"). When omitted, the
            action runs for *every* supported language so a single button-press
            on /setup-pipeline provisions both English and French-Canadian
            indexes, skillsets, data sources and indexers.
    """

    reset: bool = Field(
        default=False,
        description="If true, resets the indexer before performing the action",
    )
    language: Optional[str] = Field(
        default=None,
        description=(
            "Optional language code ('en' or 'fr-ca'). If omitted, the action "
            "is applied to all supported languages."
        ),
    )
