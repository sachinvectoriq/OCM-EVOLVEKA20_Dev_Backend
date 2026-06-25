"""Search index service for Azure AI Search."""
from abc import ABC, abstractmethod
from typing import List, Optional

from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    ComplexField,
    HnswAlgorithmConfiguration,
    HnswParameters,
    LexicalAnalyzerName,
    ScalarQuantizationCompression,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

from app.core.language import Language
from app.models.config_options import AzureOpenAIOptions


class ISearchIndexService(ABC):
    """Interface for search index service operations."""

    @abstractmethod
    async def create_search_index_async(
        self,
        index_name: str,
        include_image_processing: bool = True,
        language: Optional[str] = None,
    ) -> None:
        """Create or update a search index with vector and semantic search capabilities."""
        pass


class SearchIndexService(ISearchIndexService):
    """
    Service for managing Azure AI Search indexes.

    Handles creation and configuration of search indexes with:
    - Vector search with HNSW algorithm
    - Scalar quantization compression
    - Azure OpenAI vectorizer integration
    - Semantic search configuration
    """

    def __init__(
        self,
        index_client: SearchIndexClient,
        openai_options: AzureOpenAIOptions,
        logger,
    ) -> None:
        """
        Initialize the SearchIndexService.

        Args:
            index_client: Azure Search index client for managing indexes.
            openai_options: Configuration options for Azure OpenAI.
        """
        self._index_client: SearchIndexClient = index_client
        self._openai_options: AzureOpenAIOptions = openai_options
        self.logger = logger

    async def create_search_index_async(
        self,
        index_name: str,
        include_image_processing: bool = True,
        language: Optional[str] = None,
    ) -> None:
        """Create or update a search index with vector and semantic search capabilities.

        Applies a language-specific lexical analyzer (en.lucene / fr.lucene) to
        searchable text fields so French-Canadian content is tokenized and
        stemmed correctly. The vector configuration is language-agnostic.
        """
        lang = Language.from_value(language)
        fields = self._create_index_fields(lang)
        vector_search = self._create_vector_search_config(include_image_processing)
        semantic_search = self._create_semantic_search_config(lang)

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        await self._index_client.create_or_update_index(index)
        self.logger.info(
            f"Index '{index_name}' created or updated successfully "
            f"(language={lang.short_code}, analyzer={lang.lexical_analyzer.value})."
        )

    def _create_index_fields(self, language: Language = Language.ENGLISH) -> List[SearchField]:
        """Create the field schema for the search index, applying a language-specific analyzer."""
        analyzer = language.lexical_analyzer
        return [
            SearchField(
                name="content_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
                sortable=True,
                facetable=False,
                analyzer_name=LexicalAnalyzerName.KEYWORD,
            ),
            SearchField(
                name="text_document_id",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                sortable=False,
                facetable=False,
            ),
            SearchField(
                name="image_document_id",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                sortable=False,
                facetable=False,
            ),
            SearchField(
                name="document_title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
                analyzer_name=analyzer,
            ),
            SearchField(
                name="content_text",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
                analyzer_name=analyzer,
            ),
            SearchField(
                name="content_embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                retrievable=True,
                vector_search_dimensions=3072,
                vector_search_profile_name="hnsw",
            ),
            SearchField(
                name="content_path",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=False,
                sortable=False,
                facetable=False,
            ),
            SearchField(
                name="opco_values",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                sortable=False,
                facetable=True,
            ),
            SearchField(
                name="persona_values",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                sortable=False,
                facetable=True,
            ),
            ComplexField(
                name="location_metadata",
                fields=[
                    SimpleField(
                        name="pageNumber",
                        type=SearchFieldDataType.Int32,
                        filterable=True,
                    )
                ],
            ),
        ]

    def _create_vector_search_config(
        self, include_image_processing: bool = True
    ) -> VectorSearch:
        """
        Create vector search configuration with HNSW algorithm and compression.

        Args:
            include_image_processing: If True, uses multimodal-optimized parameters
                (m=12, ef_construction=500) for better recall with diverse content.
                If False, uses text-only parameters (m=8, ef_construction=400) for
                efficiency with homogeneous content.

        Returns:
            VectorSearch configuration object.
        """
        # Adaptive HNSW parameters based on content type
        if include_image_processing:
            # Multimodal: Higher connectivity for diverse content (text + images)
            m_value = 12
            ef_construction_value = 500
        else:
            # Text-only: Balanced settings for homogeneous content
            m_value = 8
            ef_construction_value = 400

        hnsw_config = HnswAlgorithmConfiguration(
            name="defaulthnsw",
            parameters=HnswParameters(
                m=m_value,
                ef_construction=ef_construction_value,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        )

        # Scalar quantization compression
        scalar_compression = ScalarQuantizationCompression(
            compression_name="scalar-quant-8bit"
        )

        # Azure OpenAI vectorizer
        vectorizer = AzureOpenAIVectorizer(
            vectorizer_name="multi-modal-vectorizer",
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=self._openai_options.resource_uri,
                deployment_name=self._openai_options.text_embedding_model,
                model_name=self._openai_options.text_embedding_model                      
            ),
        )

        # Vector search profile
        vector_profile = VectorSearchProfile(
            name="hnsw",
            algorithm_configuration_name="defaulthnsw",
            vectorizer_name="multi-modal-vectorizer",
            compression_name="scalar-quant-8bit",
        )

        # Initialize VectorSearch with all components
        vector_search = VectorSearch(
            algorithms=[hnsw_config],
            vectorizers=[vectorizer],
            profiles=[vector_profile],
            compressions=[scalar_compression],
        )

        return vector_search

    def _create_semantic_search_config(self, language: Language = Language.ENGLISH) -> SemanticSearch:
        """Create semantic search configuration. Naming includes the language suffix."""
        config_name = f"semanticconfig-{language.short_code}"
        semantic_config = SemanticConfiguration(
            name=config_name,
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="document_title"),
                content_fields=[
                    SemanticField(field_name="content_text")
                ],
                keywords_fields=[
                    SemanticField(field_name="opco_values"),
                    SemanticField(field_name="persona_values"),
                ],
            ),
        )

        semantic_search = SemanticSearch(
            default_configuration_name=config_name,
            configurations=[semantic_config],
        )

        return semantic_search
