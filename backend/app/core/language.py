"""Language support module for Evolve V2 multilingual pipeline.

Defines supported languages and helpers for resolving language-specific
resource names (index, skillset, data source, indexer, container) and
Azure-Search analyzer names.

Adding a new language is a matter of:
1. Adding a new enum member here.
2. Updating settings.py with corresponding env-driven defaults.
3. Adding a localized prompt entry in app/prompts/templates.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from azure.search.documents.indexes.models import LexicalAnalyzerName


class Language(str, Enum):
    """Supported content languages."""

    ENGLISH = "en"
    FRENCH_CANADIAN = "fr-ca"

    @classmethod
    def from_value(cls, value: "str | Language | None", default: "Language | None" = None) -> "Language":
        """Coerce a free-form value into a Language. Falls back to ``default`` (or English)."""
        if isinstance(value, Language):
            return value
        if value is None:
            return default or cls.ENGLISH
        v = str(value).strip().lower().replace("_", "-")
        # accept common aliases
        aliases = {
            "english": cls.ENGLISH,
            "en": cls.ENGLISH,
            "en-us": cls.ENGLISH,
            "en-ca": cls.ENGLISH,
            "fr": cls.FRENCH_CANADIAN,
            "fr-ca": cls.FRENCH_CANADIAN,
            "french": cls.FRENCH_CANADIAN,
            "french-canadian": cls.FRENCH_CANADIAN,
            "french_canadian": cls.FRENCH_CANADIAN,
            "fra": cls.FRENCH_CANADIAN,
        }
        if v in aliases:
            return aliases[v]
        try:
            return cls(v)
        except ValueError:
            return default or cls.ENGLISH

    @property
    def short_code(self) -> str:
        """Short, filesystem-safe suffix used in resource names (e.g. ``en``, ``fr``)."""
        return "en" if self is Language.ENGLISH else "fr"

    @property
    def display_name(self) -> str:
        return "English" if self is Language.ENGLISH else "French (Canadian)"

    @property
    def lexical_analyzer(self) -> LexicalAnalyzerName:
        """Azure AI Search lexical analyzer for searchable text fields."""
        if self is Language.FRENCH_CANADIAN:
            return LexicalAnalyzerName.FR_LUCENE
        return LexicalAnalyzerName.EN_LUCENE

    @property
    def document_intelligence_locale(self) -> str:
        """Locale hint for the Document Intelligence Layout skill."""
        return "fr-CA" if self is Language.FRENCH_CANADIAN else "en-US"

    @property
    def split_skill_language_code(self) -> str:
        """Language code for the SplitSkill / sentence boundary detection."""
        return "fr" if self is Language.FRENCH_CANADIAN else "en"


def all_languages() -> Iterable[Language]:
    """Yield every supported language (used for create-both pipeline runs)."""
    return tuple(Language)


def suffixed(name: str, language: Language) -> str:
    """Append a language suffix to a base resource name (idempotent).

    ``suffixed("foo-skillset", FRENCH_CANADIAN) -> "foo-skillset-fr"``
    If the name already ends with the language suffix, it's returned unchanged.
    """
    suffix = f"-{language.short_code}"
    if name.endswith(suffix):
        return name
    return f"{name}{suffix}"
