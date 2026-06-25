"""Azure AI Search service with hybrid search, semantic ranking, and filtering.

This module provides comprehensive search capabilities combining vector similarity
and keyword search with semantic ranking for RAG applications.

Features:
    - Hybrid search (keyword + vector with RRF)
    - Semantic ranking
    - Reranker score filtering (drops low-relevance results below a configurable threshold)
    - Metadata filtering
    - Result deduplication
    - Exponential backoff retry
    - Multiple search modes (vector, keyword, hybrid)

Usage:
    search_service = SearchService(search_client, openai_options)
    results = await search_service.search_async(
        query="What are Azure best practices?",
        top_k=5,
        search_mode="hybrid",
        use_semantic_ranking=True
    )
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)
from azure.core.exceptions import HttpResponseError

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from app.core.logger import Logger
from app.models import AzureOpenAIOptions, RetrievedDocument


class ISearchService(ABC):
    """Interface for search service operations."""

    @abstractmethod
    async def search_async(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        use_semantic_ranking: bool = True,
        deduplicate: bool = True,
        exclude_ids: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """Execute a search against the Azure AI Search index.

        Args:
            query: The search query text
            top_k: Number of results to return
            search_mode: 'vector', 'keyword', or 'hybrid'
            filters: Optional metadata filters
            use_semantic_ranking: Whether to use semantic ranking
            deduplicate: Whether to deduplicate results
            exclude_ids: Content IDs to exclude from results (for iterative search)

        Returns:
            List of RetrievedDocument objects ordered by relevance
        """
        pass

    @abstractmethod
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text strings.
        
        Args:
            texts: List of input texts to embed
        
        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """Get the dimensionality of generated embeddings.
        
        Returns:
            Embedding dimension count
        """
        pass


class SearchService(ISearchService):
    """Service for executing hybrid searches against Azure AI Search.

    Combines vector search (using embeddings) with full-text search
    and applies semantic ranking for optimal RAG retrieval.

    Features:
        - Hybrid search (vector + keyword)
        - Semantic ranking for improved relevance
        - Reranker score filtering: drops results below ``min_reranker_score`` when
          semantic ranking is enabled (configurable via ``SEARCHSERVICE_MIN_RERANKER_SCORE``)
        - Metadata filtering (date range, document type, category, custom)
        - Result deduplication
        - Exponential backoff retry
        - Exclusion filters for iterative search
    """

    def __init__(
        self,
        search_client: SearchClient,
        openai_options: AzureOpenAIOptions,
        logger: Logger,
        vector_field_name: str = "content_embedding",
        index_name: Optional[str] = None,
        min_reranker_score: float = 2.0,
        language: str = "en",
    ) -> None:
        """Initialize the SearchService.

        Args:
            search_client: Azure Search client for executing queries
            openai_options: Configuration for Azure OpenAI embeddings
            logger: Injected logging service
            vector_field_name: Name of the vector field in the index
            index_name: Optional index name for logging
            min_reranker_score: Minimum reranker score to retain a result when
                semantic ranking is enabled (Azure AI Search scores range 0-4)
            language: Content language ("en" or "fr-ca"). Selects the semantic
                configuration name (e.g. ``semanticconfig-en``).
        """
        from app.core.language import Language
        self._search_client: SearchClient = search_client
        self._openai_options: AzureOpenAIOptions = openai_options
        self.logger: Logger = logger
        self._openai_client: Optional[AsyncAzureOpenAI] = None
        self._vector_field_name: str = vector_field_name
        self._index_name: str = index_name or "search-index"
        self._min_reranker_score: float = min_reranker_score
        self._language: Language = Language.from_value(language)
        self._semantic_config_name: str = f"semanticconfig-{self._language.short_code}"

        self.logger.info(
            f"SearchService initialized for index: {self._index_name} "
            f"(language={self._language.short_code}, semantic_config={self._semantic_config_name})"
        )

    async def _get_openai_client(self) -> AsyncAzureOpenAI:
        """Get or create the Azure OpenAI client (lazy initialization)."""
        if self._openai_client is None:
            if self._openai_options.api_key:
                self._openai_client = AsyncAzureOpenAI(
                    azure_endpoint=self._openai_options.resource_uri,
                    api_key=self._openai_options.api_key,
                    api_version="2024-06-01",
                )
            else:
                # Use managed identity
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                self._openai_client = AsyncAzureOpenAI(
                    azure_endpoint=self._openai_options.resource_uri,
                    azure_ad_token_provider=token_provider,
                    api_version="2024-06-01",
                )
        return self._openai_client

    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text using Azure OpenAI.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            Exception: If embedding generation fails
        """
        client = await self._get_openai_client()
        
        response = await client.embeddings.create(
            input=text,
            model=self._openai_options.text_embedding_model,
        )
        
        return response.data[0].embedding

    async def search_async(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 5,
        search_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        use_semantic_ranking: bool = True,
        deduplicate: bool = True,
        exclude_ids: Optional[List[str]] = None
    ) -> List[RetrievedDocument]:
        """Execute search with vector and/or keyword components.
        
        Args:
            query: Search query text
            query_vector: Optional pre-computed query embedding
            top_k: Number of results to return
            search_mode: 'vector', 'keyword', or 'hybrid'
            filters: Optional metadata filters (date_from, date_to, document_type, category, custom, opco_values, persona_values)
            use_semantic_ranking: Whether to apply semantic ranking. When True, results
                with a reranker score below ``min_reranker_score`` are also filtered out.
            deduplicate: Whether to deduplicate results
            exclude_ids: Content IDs to exclude from results (for iterative search)
        
        Returns:
            List of retrieved documents with metadata, ordered by relevance
        """
        try:
            # Generate query embedding if not provided and needed
            if search_mode in ["vector", "hybrid"] and query_vector is None:
                query_vector = await self.generate_embedding_async(query)
            
            # Build filter expression
            if isinstance(filters, str):
                filter_expr = filters
            else:
                filter_expr = self._build_filter_expression(filters, exclude_ids)
                
            
            
            # Prepare vector query if needed
            vector_query = None
            if search_mode in ["vector", "hybrid"] and query_vector:
                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k * 2 if search_mode == "hybrid" else top_k,
                    fields=self._vector_field_name
                )
            
            # Determine search text based on mode
            search_text = query if search_mode in ["keyword", "hybrid"] else ""
            
            # Execute search with retry
            results = await self._search_with_retry(
                search_text=search_text,
                vector_queries=[vector_query] if vector_query else [],
                filter=filter_expr,
                top=top_k,
                use_semantic_ranking=use_semantic_ranking
            )
            
            # Parse results
            documents = self._parse_results(results)

            # When semantic ranking is active, drop results below the minimum reranker score
            if use_semantic_ranking:
                documents = self._filter_by_reranker_score(documents)

            # Deduplicate if requested
            if deduplicate:
                documents = self._deduplicate_results(documents)
            
            self.logger.info(
                f"Search ({search_mode}) returned {len(documents)} documents for query: {query[:50]}..."
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}", exc_info=True)
            raise

    async def _search_with_retry(
        self,
        search_text: str,
        vector_queries: List[VectorizedQuery],
        filter: Optional[str],
        top: int,
        use_semantic_ranking: bool,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Execute search with exponential backoff retry.
        
        Args:
            search_text: Keyword search query
            vector_queries: Vector search queries
            filter: OData filter expression
            top: Number of results
            use_semantic_ranking: Whether to enable semantic ranking (QueryType.SEMANTIC)
            max_retries: Maximum retry attempts
            base_delay: Base delay in seconds for exponential backoff
        
        Returns:
            Raw search results as a list of result dicts from Azure AI Search
        """
        for attempt in range(max_retries + 1):
            try:
                # Build search parameters
                search_params = {
                    "search_text": search_text,
                    "vector_queries": vector_queries,
                    "filter": filter,
                    "top": top,
                    "select": ["content_id", "text_document_id", "image_document_id", "document_title", "content_text", "content_path", "location_metadata"]
                }
                
                # Add semantic ranking if enabled
                if use_semantic_ranking:
                    search_params.update({
                        "query_type": QueryType.SEMANTIC,
                        "semantic_configuration_name": self._semantic_config_name,
                        "query_caption": QueryCaptionType.EXTRACTIVE,
                        "query_answer": QueryAnswerType.EXTRACTIVE
                    })
                
                # Execute search
                response = await self._search_client.search(**search_params)
                
                # Collect results
                results = []
                async for result in response:
                    results.append(result)
                
                return results
                
            except HttpResponseError as e:
                if e.status_code == 429 and attempt < max_retries:
                    # Rate limited - retry with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Search rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            
            except Exception as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Search failed, retrying in {delay}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        
        raise RuntimeError(f"Search failed after {max_retries} retries")

    def _filter_by_reranker_score(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Drop documents whose reranker score is below the minimum threshold.

        Only documents that actually carry a reranker score are filtered; documents
        without one (e.g. when semantic ranking was unexpectedly absent) are kept.

        Args:
            documents: Parsed search results.

        Returns:
            Filtered list with low-relevance results removed.
        """
        filtered = [
            doc for doc in documents
            if doc.reranker_score is None or doc.reranker_score >= self._min_reranker_score
        ]
        dropped = len(documents) - len(filtered)
        if dropped:
            self.logger.info(
                f"[RerankerFilter] Dropped {dropped}/{len(documents)} results "
                f"with reranker_score < {self._min_reranker_score}"
            )
        return filtered

    def _build_filter_expression(
        self, 
        filters: Optional[Dict[str, Any]],
        exclude_ids: Optional[List[str]] = None
    ) -> Optional[str]:
        """Build OData filter expression from filter dictionary.
        
        Supported filters:
        - date_from/date_to: Date range filtering
        - document_type: Document type filter
        - category: Category filter
        - opco_values: Operating company filter (collection field)
        - persona_values: Persona filter (collection field)
        - custom: Any custom OData expression
        
        Args:
            filters: Dictionary of filter criteria
            exclude_ids: Content IDs to exclude from results
        
        Returns:
            OData filter expression or None
        """
        filter_parts = []
        
        if filters:
            # Date range filtering
            if "date_from" in filters:
                filter_parts.append(f"metadata/date ge {filters['date_from']}")
            
            if "date_to" in filters:
                filter_parts.append(f"metadata/date le {filters['date_to']}")
            
            # Document type filtering
            if "document_type" in filters:
                doc_types = filters["document_type"]
                if isinstance(doc_types, list):
                    type_filters = " or ".join(
                        f"metadata/documentType eq '{dt}'" for dt in doc_types
                    )
                    filter_parts.append(f"({type_filters})")
                else:
                    filter_parts.append(f"metadata/documentType eq '{doc_types}'")
            
            # Category filtering
            if "category" in filters:
                categories = filters["category"]
                if isinstance(categories, list):
                    category_filters = " or ".join(
                        f"metadata/category eq '{cat}'" for cat in categories
                    )
                    filter_parts.append(f"({category_filters})")
                else:
                    filter_parts.append(f"metadata/category eq '{categories}'")
            
            # Operating company filtering (collection field)
            if "opco_values" in filters:
                opco_values = filters["opco_values"]
                if isinstance(opco_values, list):
                    opco_conditions = " or ".join(
                        f"opco_values/any(o: o eq '{value}')" for value in opco_values
                    )
                    filter_parts.append(f"({opco_conditions})")
            
            # Persona filtering (collection field)
            if "persona_values" in filters:
                persona_values = filters["persona_values"]
                if isinstance(persona_values, list):
                    persona_conditions = " or ".join(
                        f"persona_values/any(p: p eq '{value}')" for value in persona_values
                    )
                    filter_parts.append(f"({persona_conditions})")
            
            # Custom OData expression
            if "custom" in filters:
                filter_parts.append(filters["custom"])
        
        # Add exclusion filter for already processed documents
        if exclude_ids:
            # Escape single quotes in IDs and join with commas
            escaped_ids = [id.replace("'", "''") for id in exclude_ids]
            excluded_ids_str = ','.join(escaped_ids)
            filter_parts.append(f"not search.in(content_id, '{excluded_ids_str}', ',')")
        
        return " and ".join(filter_parts) if filter_parts else None

    def _parse_results(self, results: List[Dict[str, Any]]) -> List[RetrievedDocument]:
        """Parse search results into RetrievedDocument objects.
        
        Args:
            results: Raw search results from Azure AI Search
        
        Returns:
            List of parsed documents
        """
        documents = []
        
        for result in results:
            try:
                # Extract document ID from base64 encoded text_document_id or image_document_id
                # Prefer text_document_id, fall back to image_document_id
                document_id = result.get("text_document_id") or result.get("image_document_id") or ""
                
                doc = RetrievedDocument(
                    document_id=document_id,
                    content_id=result.get("content_id") or "",
                    title=result.get("document_title") or "",
                    content=result.get("content_text") or "",
                    source=result.get("content_path") or "",
                    page_number=result.get("location_metadata", {}).get("pageNumber"),
                    score=result.get("@search.score", 0.0),
                    reranker_score=result.get("@search.reranker_score"),
                    metadata={}
                )

                # TODO: don't think this is needed, but leaving for now
                # Add semantic captions if available
                #if "@search.captions" in result:
                #    captions = result["@search.captions"]
                #    if captions:
                #        doc.metadata["semantic_caption"] = captions[0].get("text", "")
                
                documents.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse search result: {e}", exc_info=True)
                continue
        
        return documents

    def _deduplicate_results(
        self,
        documents: List[RetrievedDocument],
        similarity_threshold: float = 0.95
    ) -> List[RetrievedDocument]:
        """Deduplicate search results.
        
        Deduplication strategies:
        1. Remove exact content ID duplicates (unique chunks)
        2. Remove near-duplicate content (high similarity)
        
        Args:
            documents: List of retrieved documents
            similarity_threshold: Content similarity threshold for deduplication
        
        Returns:
            Deduplicated list of documents
        """
        if not documents:
            return documents
        
        # Track seen content IDs (unique chunks)
        seen_ids = set()
        deduplicated = []
        
        for doc in documents:
            # Skip exact content ID duplicates
            if doc.content_id in seen_ids:
                self.logger.debug(f"Skipping duplicate content ID: {doc.content_id}")
                continue
            
            # Check content similarity with existing documents
            is_duplicate = False
            for existing_doc in deduplicated:
                similarity = self._calculate_content_similarity(
                    doc.content,
                    existing_doc.content
                )
                
                if similarity >= similarity_threshold:
                    self.logger.debug(
                        f"Skipping near-duplicate content (similarity: {similarity:.2f})"
                    )
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_ids.add(doc.content_id)
                deduplicated.append(doc)
        
        if len(deduplicated) < len(documents):
            self.logger.info(
                f"Deduplication: {len(documents)} -> {len(deduplicated)} documents"
            )
        
        return deduplicated

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (Jaccard similarity).
        
        Args:
            content1: First content string
            content2: Second content string
        
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word-based Jaccard similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text strings.
        
        Args:
            texts: List of input texts to embed
        
        Returns:
            List of embedding vectors
        """
        try:
            self.logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            # Get OpenAI client
            client = await self._get_openai_client()
            
            # Call embedding API with batch of texts
            response = await client.embeddings.create(
                input=texts,
                model=self._openai_options.text_embedding_model,
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            self.logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
            raise
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensionality of generated embeddings.
        
        Returns:
            Embedding dimension count (typically 1536 for text-embedding-ada-002,
            or 3072 for text-embedding-3-large)
        """
        # Common embedding dimensions by model
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        model_name = self._openai_options.text_embedding_model
        return model_dimensions.get(model_name, 1536)  # Default to 1536

    async def close(self):
        """Close the search client."""
        await self._search_client.close()
