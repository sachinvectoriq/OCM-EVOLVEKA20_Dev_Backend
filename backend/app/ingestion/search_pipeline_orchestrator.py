"""Search pipeline orchestrator for coordinating multimodal indexing."""
from abc import ABC, abstractmethod
from typing import Optional, Iterable

from azure.core.exceptions import ResourceNotFoundError

from app.core.language import Language, all_languages
from app.models.config_options import SearchServiceOptions, BlobStorageOptions
from app.ingestion.data_source_service import IDataSourceService
from app.ingestion.indexer_service import IIndexerService
from app.ingestion.search_index_service import ISearchIndexService
from app.ingestion.skillset_service import ISkillsetService


class ISearchPipelineOrchestrator(ABC):
    """Interface for search pipeline orchestration operations."""

    @abstractmethod
    async def setup_pipeline_async(self, language: Optional[str] = None) -> None:
        """Set up the complete search pipeline for one language."""
        pass

    @abstractmethod
    async def setup_all_languages_async(
        self, languages: Optional[Iterable[str]] = None
    ) -> None:
        """Set up the complete search pipeline for every supported language.

        Default behaviour creates two pipelines (English and French-Canadian)
        each with its own data source, index, skillset and indexer.
        """
        pass

    @abstractmethod
    async def run_indexer_async(self, language: Optional[str] = None) -> None:
        """Run the indexer for a given language (defaults to English)."""
        pass

    @abstractmethod
    async def is_first_run_async(self, language: Optional[str] = None) -> bool:
        """Check if this is the first run of the pipeline for the given language."""
        pass


class SearchPipelineOrchestrator(ISearchPipelineOrchestrator):
    """Orchestrates the Azure AI Search multimodal pipeline (per language)."""

    def __init__(
        self,
        data_source_service: IDataSourceService,
        search_index_service: ISearchIndexService,
        skillset_service: ISkillsetService,
        indexer_service: IIndexerService,
        search_options: SearchServiceOptions,
        logger,
        settings=None,
    ) -> None:
        self._data_source_service: IDataSourceService = data_source_service
        self._search_index_service: ISearchIndexService = search_index_service
        self._skillset_service: ISkillsetService = skillset_service
        self._indexer_service: IIndexerService = indexer_service
        self._search_options: SearchServiceOptions = search_options
        self._settings = settings
        self.logger = logger

    # ------------------------------------------------------------------
    # Helpers: resolve per-language search/blob options from settings
    # ------------------------------------------------------------------
    def _options_for(self, language: Language) -> SearchServiceOptions:
        if self._settings is None:
            # Fallback: only English supported when settings aren't provided.
            return self._search_options
        return self._settings.search_service_options_for(language.value)

    def _blob_for(self, language: Language) -> Optional[BlobStorageOptions]:
        if self._settings is None:
            return None
        return self._settings.blob_storage_options_for(language.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def setup_pipeline_async(self, language: Optional[str] = None) -> None:
        """Provision data source, index, skillset, and indexer for one language."""
        lang = Language.from_value(language)
        opts = self._options_for(lang)
        blob_opts = self._blob_for(lang)

        self.logger.info(
            f"Setting up search pipeline for language={lang.short_code} "
            f"(index={opts.index_name}, skillset={opts.skillset_name})..."
        )

        # 1. Data source — points at the language-specific blob container
        await self._data_source_service.create_blob_data_source_async(
            opts.data_source_name, blob_options=blob_opts
        )
        self.logger.info(f"[{lang.short_code}] Data source created: {opts.data_source_name}")

        # 2. Index with language-specific analyzer
        await self._search_index_service.create_search_index_async(
            opts.index_name, language=lang.value
        )
        self.logger.info(f"[{lang.short_code}] Index created: {opts.index_name}")

        # 3. Skillset with language-specific prompts
        await self._skillset_service.create_skillset_using_sdk_async(
            opts.skillset_name,
            opts.index_name,
            language=lang.value,
            blob_options=blob_opts,
        )
        self.logger.info(f"[{lang.short_code}] Skillset created: {opts.skillset_name}")

        # 4. Indexer
        await self._indexer_service.create_indexer_async(
            opts.indexer_name,
            opts.data_source_name,
            opts.index_name,
            opts.skillset_name,
        )
        self.logger.info(f"[{lang.short_code}] Indexer created: {opts.indexer_name}")
        self.logger.info(f"Search pipeline setup complete for language={lang.short_code}.")

    async def setup_all_languages_async(
        self, languages: Optional[Iterable[str]] = None
    ) -> None:
        """Provision a full pipeline (data source + index + skillset + indexer) for each language."""
        targets: list[Language] = (
            [Language.from_value(l) for l in languages]
            if languages
            else list(all_languages())
        )
        self.logger.info(
            f"Setting up pipelines for languages: {[l.short_code for l in targets]}"
        )
        for lang in targets:
            await self.setup_pipeline_async(language=lang.value)

    async def run_indexer_async(self, language: Optional[str] = None) -> None:
        lang = Language.from_value(language)
        opts = self._options_for(lang)
        self.logger.info(f"Running indexer '{opts.indexer_name}' for language={lang.short_code}...")
        await self._indexer_service.run_indexer_async(opts.indexer_name)
        self.logger.info("Indexer run initiated.")

    async def is_first_run_async(self, language: Optional[str] = None) -> bool:
        lang = Language.from_value(language)
        opts = self._options_for(lang)
        try:
            await self._indexer_service.get_indexer_status_async(opts.indexer_name)
            self.logger.info(f"Indexer exists for language={lang.short_code}. Not first run.")
            return False
        except ResourceNotFoundError:
            self.logger.info(f"Indexer not found for language={lang.short_code}. This is the first run.")
            return True
