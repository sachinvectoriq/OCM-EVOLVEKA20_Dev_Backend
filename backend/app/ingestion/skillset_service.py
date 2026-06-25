"""Skillset service for Azure AI Search."""

import httpx
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import timedelta

from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.models import (
    AIServicesAccountIdentity,
    AzureOpenAIEmbeddingSkill,
    ChatCompletionSkill,
    DocumentIntelligenceLayoutSkill,
    DocumentIntelligenceLayoutSkillChunkingProperties,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerKnowledgeStore,
    SearchIndexerKnowledgeStoreObjectProjectionSelector,
    SearchIndexerKnowledgeStoreProjection,
    SearchIndexerSkillset,
    ShaperSkill,
    SplitSkill,
    WebApiSkill,
)

from app.core.language import Language
from app.models.config_options import (
    AIServicesOptions,
    AzureOpenAIOptions,
    BlobStorageOptions,
    SearchServiceOptions,
)
from app.prompts.templates import IngestionPrompts
from app.prompts.localized import get_ingestion_prompts


class ISkillsetService(ABC):
    async def create_skillset_using_sdk_async(
        self,
        skillset_name: str,
        index_name: str,
        language: Optional[str] = None,
        blob_options: Optional[BlobStorageOptions] = None,
    ) -> None:
        pass

    async def create_skillset_using_rest_async(
        self,
        skillset_name: str,
        index_name: str,
        language: Optional[str] = None,
        blob_options: Optional[BlobStorageOptions] = None,
    ) -> None:
        pass


class SkillsetService(ISkillsetService):
    # English defaults preserved for backwards compatibility with any code path
    # that doesn't pass a language. Use `_literals_for_language` at runtime to
    # get the localized escaped string literals.
    IMAGE_VERBALIZATION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.IMAGE_VERBALIZATION_SYSTEM_MESSAGE
    )
    OPCO_EXTRACTION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.OPCO_EXTRACTION_SYSTEM_MESSAGE
    )
    PERSONA_EXTRACTION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.PERSONA_EXTRACTION_SYSTEM_MESSAGE
    )
    OPCO_NORMALIZATION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.OPCO_VALUE_NORMALIZATION_SYSTEM_MESSAGE
    )
    PERSONA_NORMALIZATION_LITERAL = IngestionPrompts.as_search_string_literal(
        IngestionPrompts.PERSONA_VALUE_NORMALIZATION_SYSTEM_MESSAGE
    )

    def __init__(
        self,
        search_indexer_client: SearchIndexerClient,
        search_options: SearchServiceOptions,
        openai_options: AzureOpenAIOptions,
        ai_services_options: AIServicesOptions,
        blob_options: BlobStorageOptions,
        logger,
    ) -> None:
        self._search_indexer_client: SearchIndexerClient = search_indexer_client
        self._search_options: SearchServiceOptions = search_options
        self._openai_options: AzureOpenAIOptions = openai_options
        self._ai_services_options: AIServicesOptions = ai_services_options
        self._blob_options: BlobStorageOptions = blob_options
        self.logger = logger

    def _literals_for_language(self, language: Optional[str]):
        """Return Azure-Search escaped string literals for prompts in the given language."""
        bundle = get_ingestion_prompts(language)
        return {
            "image": IngestionPrompts.as_search_string_literal(bundle.image_verbalization),
            "opco_extract": IngestionPrompts.as_search_string_literal(bundle.opco_extraction),
            "persona_extract": IngestionPrompts.as_search_string_literal(bundle.persona_extraction),
            "opco_norm": IngestionPrompts.as_search_string_literal(bundle.opco_normalization),
            "persona_norm": IngestionPrompts.as_search_string_literal(bundle.persona_normalization),
        }

    _CHAT_COMPLETION_API_VERSION_FALLBACK = "2025-04-01-preview"

    def _ensure_chat_completion_api_version(self, uri: str) -> str:
        """Ensure the chat completion URL has an api-version query parameter.

        The api-version is always sourced from ``AzureOpenAIOptions.chat_completion_api_version``
        (which comes from the ``AZURE_OPENAI_API_VERSION`` env var). This handles the case where
        PowerShell .env loaders split on '=' and truncate the URL's trailing
        ``?api-version=<value>`` part.
        """
        api_version = (
            self._openai_options.chat_completion_api_version
            or self._CHAT_COMPLETION_API_VERSION_FALLBACK
        )
        if not uri:
            return uri
        # Dangling "?api-version" or "&api-version" (no '=')
        if uri.endswith("?api-version") or uri.endswith("&api-version"):
            return f"{uri}={api_version}"
        # Dangling "...api-version=" with empty value
        if uri.endswith("api-version="):
            return f"{uri}{api_version}"
        # Already has api-version=<something> — leave untouched
        if "api-version=" in uri:
            return uri
        # No api-version present — append
        sep = "&" if "?" in uri else "?"
        return f"{uri}{sep}api-version={api_version}"

    async def create_skillset_using_sdk_async(
        self,
        skillset_name: str,
        index_name: str,
        language: Optional[str] = None,
        blob_options: Optional[BlobStorageOptions] = None,
    ) -> None:
        lang = Language.from_value(language)
        literals = self._literals_for_language(lang)
        blob_opts = blob_options or self._blob_options
        skills: List = [
            DocumentIntelligenceLayoutSkill(
                name="document-intelligence-layout-skill",
                description=(
                    f"Extract text and images with layout from documents using "
                    f"Document Intelligence (locale: {lang.document_intelligence_locale})"
                ),
                context="/document",
                output_mode="oneToMany",
                output_format="text",
                markdown_header_depth=None,
                extraction_options=["images", "locationMetadata"],
                chunking_properties=DocumentIntelligenceLayoutSkillChunkingProperties(
                    unit="characters",
                    maximum_length=3000,
                    overlap_length=500,
                ),
                inputs=[
                    InputFieldMappingEntry(name="file_data", source="/document/file_data")
                ],
                outputs=[
                    OutputFieldMappingEntry(name="text_sections", target_name="text_sections"),
                    OutputFieldMappingEntry(name="normalized_images", target_name="normalized_images"),
                ],
            ),
            AzureOpenAIEmbeddingSkill(
                name="text-chunk-embedding-skill",
                description="Generate embeddings for text chunks using Azure OpenAI",
                context="/document/text_sections/*",
                resource_url=self._openai_options.resource_uri,
                deployment_name=self._openai_options.text_embedding_model,
                dimensions=3072,
                model_name=self._openai_options.text_embedding_model,
                inputs=[
                    InputFieldMappingEntry(
                        name="text",
                        source="/document/text_sections/*/content",
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(name="embedding", target_name="text_vector")
                ],
            ),
            WebApiSkill(
                name="blob-metadata-enrichment-skill",
                description="Parse opco/persona strings pulled from blob metadata by the indexer",
                context="/document",
                uri="https://app-ka-sandbox-001.azurewebsites.net/api/enrich",
                http_method="POST",
                timeout=timedelta(seconds=30),
                batch_size=10,
                inputs=[
                    InputFieldMappingEntry(
                        name="opco",
                        source="/document/opco",
                    ),
                    InputFieldMappingEntry(
                        name="persona",
                        source="/document/persona",
                    ),
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="opco_values",
                        target_name="opco_values_array",
                    ),
                    OutputFieldMappingEntry(
                        name="persona_values",
                        target_name="persona_values_array",
                    ),
                ],
            ),
            ChatCompletionSkill(
                name="image-verbalization-skill",
                description="Generate text descriptions of images using Azure OpenAI chat completions (vision)",
                context="/document/normalized_images/*",
                uri=self._ensure_chat_completion_api_version(self._openai_options.effective_chat_completion_uri),
                auth_resource_id="https://cognitiveservices.azure.com",
                http_method="POST",
                timeout=timedelta(minutes=3, seconds=50),
                batch_size=1,
                inputs=[
                    InputFieldMappingEntry(
                        name="systemMessage",
                        source=literals["image"],
                    ),
                    InputFieldMappingEntry(
                        name="userMessage",
                        source="='Please describe this image.'" if lang is Language.ENGLISH else "='Veuillez décrire cette image.'",
                    ),
                    InputFieldMappingEntry(
                        name="image",
                        source="/document/normalized_images/*/data",
                    ),
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="response",
                        target_name="verbalizedImage"
                    ),
                ],
            ),
            AzureOpenAIEmbeddingSkill(
                name="image-description-embedding-skill",
                description="Generate embeddings for image descriptions using Azure OpenAI",
                context="/document/normalized_images/*",
                resource_url=self._openai_options.resource_uri,
                deployment_name=self._openai_options.text_embedding_model,
                dimensions=3072,
                model_name=self._openai_options.text_embedding_model,
                inputs=[
                    InputFieldMappingEntry(
                        name="text",
                        source="/document/normalized_images/*/verbalizedImage",
                    )
                ],
                outputs=[
                    OutputFieldMappingEntry(
                        name="embedding",
                        target_name="verbalizedImage_vector",
                    )
                ],
            ),
            ShaperSkill(
                name="image-path-shaper-skill",
                context="/document/normalized_images/*",
                inputs=[
                    InputFieldMappingEntry(
                        name="normalized_images",
                        source="/document/normalized_images/*",
                    ),
                    InputFieldMappingEntry(
                        name="imagePath",
                        source=f"='{blob_opts.images_container_name}/' + $(/document/normalized_images/*/imagePath)",
                    ),
                ],
                outputs=[
                    OutputFieldMappingEntry(name="output", target_name="new_normalized_images")
                ],
            ),
        ]

        projection_selectors = [
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="text_document_id",
                source_context="/document/text_sections/*",
                mappings=[
                    InputFieldMappingEntry(
                        name="content_embedding",
                        source="/document/text_sections/*/text_vector",
                    ),
                    InputFieldMappingEntry(
                        name="content_text",
                        source="/document/text_sections/*/content",
                    ),
                    InputFieldMappingEntry(
                        name="location_metadata",
                        source="/document/text_sections/*/locationMetadata",
                    ),
                    InputFieldMappingEntry(
                        name="document_title",
                        source="/document/document_title",
                    ),
                    InputFieldMappingEntry(
                        name="opco_values",
                        source="/document/opco_values_array/*",
                    ),
                    InputFieldMappingEntry(
                        name="persona_values",
                        source="/document/persona_values_array/*",
                    ),
                ],
            ),
            SearchIndexerIndexProjectionSelector(
                target_index_name=index_name,
                parent_key_field_name="image_document_id",
                source_context="/document/normalized_images/*",
                mappings=[
                    InputFieldMappingEntry(
                        name="content_text",
                        source="/document/normalized_images/*/verbalizedImage",
                    ),
                    InputFieldMappingEntry(
                        name="content_embedding",
                        source="/document/normalized_images/*/verbalizedImage_vector",
                    ),
                    InputFieldMappingEntry(
                        name="content_path",
                        source="/document/normalized_images/*/new_normalized_images/imagePath",
                    ),
                    InputFieldMappingEntry(
                        name="document_title",
                        source="/document/document_title",
                    ),
                    InputFieldMappingEntry(
                        name="location_metadata",
                        source="/document/normalized_images/*/locationMetadata",
                    ),
                    InputFieldMappingEntry(
                        name="opco_values",
                        source="/document/opco_values_array/*",
                    ),
                    InputFieldMappingEntry(
                        name="persona_values",
                        source="/document/persona_values_array/*",
                    ),
                ],
            ),
        ]

        index_projections = SearchIndexerIndexProjection(
            selectors=projection_selectors,
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

        knowledge_store = SearchIndexerKnowledgeStore(
            storage_connection_string=f"ResourceId={blob_opts.resource_id}",
            projections=[
                SearchIndexerKnowledgeStoreProjection(
                    objects=[
                        SearchIndexerKnowledgeStoreObjectProjectionSelector(
                            storage_container=blob_opts.images_container_name,
                            source="/document/normalized_images/*",
                        )
                    ]
                )
            ],
            parameters={"synthesizeGeneratedKeyName": True},
        )

        endpoint_str = str(self._ai_services_options.cognitive_services_endpoint).rstrip("/")

        cognitive_services_account = AIServicesAccountIdentity(
            subdomain_url=endpoint_str,
        )

        skillset = SearchIndexerSkillset(
            name=skillset_name,
            description=(
                f"Skillset for multimodal document processing "
                f"({lang.display_name}, locale {lang.document_intelligence_locale})"
            ),
            skills=skills,
            cognitive_services_account=cognitive_services_account,
            index_projection=index_projections,
            knowledge_store=knowledge_store,
        )

        await self._search_indexer_client.create_or_update_skillset(skillset)

    async def create_skillset_using_rest_async(
        self,
        skillset_name: str,
        index_name: str,
        language: Optional[str] = None,
        blob_options: Optional[BlobStorageOptions] = None,
    ) -> None:
        lang = Language.from_value(language)
        literals = self._literals_for_language(lang)
        blob_opts = blob_options or self._blob_options
        endpoint_str = str(self._ai_services_options.cognitive_services_endpoint).rstrip("/")

        cognitive_services = {
            "@odata.type": "#Microsoft.Azure.Search.AIServicesByIdentity",
            "subdomainUrl": endpoint_str,
        }

        skills: List[Dict[str, Any]] = [
            {
                "@odata.type": "#Microsoft.Skills.Util.DocumentIntelligenceLayoutSkill",
                "name": "document-intelligence-layout-skill",
                "description": "Extract text and images with layout from documents using Document Intelligence",
                "context": "/document",
                "outputMode": "oneToMany",
                "outputFormat": "text",
                "extractionOptions": ["images", "locationMetadata"],
                "chunkingProperties": {
                    "unit": "characters",
                    "maximumLength": 3000,
                    "overlapLength": 500,
                },
                "inputs": [
                    {"name": "file_data", "source": "/document/file_data"}
                ],
                "outputs": [
                    {"name": "text_sections", "targetName": "text_sections"},
                    {"name": "normalized_images", "targetName": "normalized_images"},
                ],
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "name": "text-chunk-embedding-skill",
                "description": "Generate embeddings for text chunks using Azure OpenAI",
                "context": "/document/text_sections/*",
                "resourceUri": self._openai_options.resource_uri,
                "deploymentId": self._openai_options.text_embedding_model,
                "dimensions": 3072,
                "modelName": self._openai_options.text_embedding_model,
                "inputs": [
                    {"name": "text", "source": "/document/text_sections/*/content"}
                ],
                "outputs": [
                    {"name": "embedding", "targetName": "text_vector"}
                ],
            },
            {
                "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
                "name": "blob-metadata-enrichment-skill",
                "description": "Parse opco/persona strings pulled from blob metadata by the indexer",
                "context": "/document",
                "uri": "https://app-ka-sandbox-001.azurewebsites.net/api/enrich",
                "httpMethod": "POST",
                "timeout": "PT30S",
                "batchSize": 10,
                "inputs": [
                    {"name": "opco", "source": "/document/opco"},
                    {"name": "persona", "source": "/document/persona"},
                ],
                "outputs": [
                    {"name": "opco_values", "targetName": "opco_values_array"},
                    {"name": "persona_values", "targetName": "persona_values_array"},
                ],
                "httpHeaders": {},
            },
            {
                "@odata.type": "#Microsoft.Skills.Custom.ChatCompletionSkill",
                "name": "image-verbalization-skill",
                "description": "Generate text descriptions of images using Azure OpenAI chat completions (vision)",
                "context": "/document/normalized_images/*",
                "uri": self._ensure_chat_completion_api_version(self._openai_options.effective_chat_completion_uri),
                "authResourceId": "https://cognitiveservices.azure.com",
                "httpMethod": "POST",
                "timeout": "PT3M50S",
                "batchSize": 1,
                "inputs": [
                    {"name": "systemMessage", "source": literals["image"]},
                    {"name": "userMessage", "source": ("='Please describe this image.'" if lang is Language.ENGLISH else "='Veuillez décrire cette image.'")},
                    {"name": "image", "source": "/document/normalized_images/*/data"},
                ],
                "outputs": [
                    {"name": "response", "targetName": "verbalizedImage"}
                ],
                "httpHeaders": {},
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "name": "image-description-embedding-skill",
                "description": "Generate embeddings for image descriptions using Azure OpenAI",
                "context": "/document/normalized_images/*",
                "resourceUri": self._openai_options.resource_uri,
                "deploymentId": self._openai_options.text_embedding_model,
                "dimensions": 3072,
                "modelName": self._openai_options.text_embedding_model,
                "inputs": [
                    {"name": "text", "source": "/document/normalized_images/*/verbalizedImage"}
                ],
                "outputs": [
                    {"name": "embedding", "targetName": "verbalizedImage_vector"}
                ],
            },
            {
                "@odata.type": "#Microsoft.Skills.Util.ShaperSkill",
                "name": "image-path-shaper-skill",
                "context": "/document/normalized_images/*",
                "inputs": [
                    {"name": "normalized_images", "source": "/document/normalized_images/*"},
                    {
                        "name": "imagePath",
                        "source": f"='{blob_opts.images_container_name}/' + $(/document/normalized_images/*/imagePath)",
                    },
                ],
                "outputs": [
                    {"name": "output", "targetName": "new_normalized_images"}
                ],
            },
        ]

        selectors: List[Dict[str, Any]] = [
            {
                "targetIndexName": index_name,
                "parentKeyFieldName": "text_document_id",
                "sourceContext": "/document/text_sections/*",
                "mappings": [
                    {"name": "content_embedding", "source": "/document/text_sections/*/text_vector"},
                    {"name": "content_text", "source": "/document/text_sections/*/content"},
                    {"name": "location_metadata", "source": "/document/text_sections/*/locationMetadata"},
                    {"name": "document_title", "source": "/document/document_title"},
                    {"name": "opco_values", "source": "/document/opco_values_array/*"},
                    {"name": "persona_values", "source": "/document/persona_values_array/*"},
                ],
            },
            {
                "targetIndexName": index_name,
                "parentKeyFieldName": "image_document_id",
                "sourceContext": "/document/normalized_images/*",
                "mappings": [
                    {"name": "content_text", "source": "/document/normalized_images/*/verbalizedImage"},
                    {"name": "content_embedding", "source": "/document/normalized_images/*/verbalizedImage_vector"},
                    {"name": "content_path", "source": "/document/normalized_images/*/new_normalized_images/imagePath"},
                    {"name": "document_title", "source": "/document/document_title"},
                    {"name": "location_metadata", "source": "/document/normalized_images/*/locationMetadata"},
                    {"name": "opco_values", "source": "/document/opco_values_array/*"},
                    {"name": "persona_values", "source": "/document/persona_values_array/*"},
                ],
            },
        ]

        index_projections: Dict[str, Any] = {
            "selectors": selectors,
            "parameters": {"projectionMode": "skipIndexingParentDocuments"},
        }

        payload: Dict[str, Any] = {
            "name": skillset_name,
            "description": "A skillset for multimodal document processing with text and image extraction",
            "cognitiveServices": cognitive_services,
            "skills": skills,
            "indexProjections": index_projections,
        }

        storage_arm_id = blob_opts.resource_id
        if isinstance(storage_arm_id, str) and storage_arm_id.startswith("ResourceId="):
            storage_arm_id = storage_arm_id.split("ResourceId=", 1)[1]

        payload["knowledgeStore"] = {
            "storageConnectionString": f"ResourceId={storage_arm_id}",
            "projections": [
                {
                    "objects": [
                        {
                            "storageContainer": blob_opts.images_container_name,
                            "source": "/document/normalized_images/*",
                        }
                    ]
                }
            ],
            "parameters": {"synthesizeGeneratedKeyName": True},
        }

        endpoint = str(self._search_options.endpoint).rstrip("/")
        url = f"{endpoint}/skillsets/{skillset_name}"
        params = {"api-version": self._search_options.skillset_api_version}

        headers = {
            "api-key": self._search_options.api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            resp = await client.put(url, params=params, headers=headers, json=payload)

        if 200 <= resp.status_code < 300:
            return

        raise Exception(
            f"Error creating skillset '{skillset_name}' via REST: {resp.status_code} - {resp.text}"
        )