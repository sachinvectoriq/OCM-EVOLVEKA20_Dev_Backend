"""Dependency injection container for Knowledge Assistant services.

This module defines the DI container that manages service lifecycle and dependency wiring using dependency-injector.

Purpose:    
    - Wire up service dependencies with automatic injection
    - Provide application configuration to all components
    - Enable testability with mock providers for unit tests
    - Ensure proper resource lifecycle management

Usage in FastAPI (main.py):
    ```python
    from app.core.container import Container
    
    # Initialize container
    container = Container()
    
    # Wire routes for dependency injection
    container.wire(modules=[
        "app.api.routes.chat",
        "app.api.routes.health",
        "app.api.routes.admin"
    ])
    
    # Attach to FastAPI app
    app = FastAPI()
    app.container = container
    ```
    
Testing with Mocks:
    ```python
    from dependency_injector import providers
    
    container = Container()
    container.cosmos_client.override(providers.Singleton(MockCosmosClient))
    container.search_service.override(providers.Singleton(MockSearchService))
    ```

Provider Types and When to Use:
    
    Singleton - Single instance shared across all requests
        Use for:
        ✓ Azure SDK clients (CosmosClient, SearchClient)
        ✓ Configuration (Settings) and Logger
        ✓ Stateless services that don't maintain request-specific state
        ✓ Services with expensive initialization (connection pooling)
        Benefits: Reuse connection pools, minimize overhead, better performance
    
    Factory - New instance created for each injection
        Use for:
        ✓ Stateful services that maintain request-specific data (agents)
        ✓ Services that need request isolation (chat stores per thread)
        ✓ Testing when fresh instances are needed per test
        Benefits: Isolation, no shared state, thread-safety for stateful objects
"""

from dependency_injector import containers, providers
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient, SearchIndexerClient
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential

from app.core.settings import Settings
from app.core.logger import create_logger, Logger
from app.models import (
    SearchServiceOptions,
    BlobStorageOptions,
    AIServicesOptions,
    AzureOpenAIOptions,
    AzureAIFoundryOptions,
    CosmosDBOptions,
    KeyVaultOptions,
    ApplicationInsightsOptions,
    WorkflowOptions,
    APIOptions,
)
from app.ingestion.data_source_service import IDataSourceService, DataSourceService
from app.ingestion.search_index_service import ISearchIndexService, SearchIndexService
from app.ingestion.skillset_service import ISkillsetService, SkillsetService
from app.ingestion.indexer_service import IIndexerService, IndexerService
from app.ingestion.search_pipeline_orchestrator import ISearchPipelineOrchestrator, SearchPipelineOrchestrator
from app.services.search_service import ISearchService, SearchService
from app.services.chat_service import IChatService, ChatService
from app.services.conversation_service import IConversationService, ConversationService
from app.agents.answer_generator import AnswerGenerator
from app.agents.query_rewriter import QueryRewriter
from app.agents.reflection_agent import ReflectionAgent
from app.utils.citation_tracker import CitationTracker
from app.workflows.agentic_rag_workflow import AgenticRAGWorkflow


def _create_cosmos_client(options: CosmosDBOptions) -> CosmosClient:
    """Create a Cosmos client based on available configuration."""
    if options.connection_string:
        return CosmosClient.from_connection_string(options.connection_string)
    elif options.endpoint:
        return CosmosClient(options.endpoint, credential=DefaultAzureCredential())
    else:
        raise ValueError("CosmosDBOptions must include either connection_string or endpoint.")


# Pin to a Search REST API version supported by both the service and the
# installed azure-search-documents beta SDK. The SDK's default
# (2025-11-01-preview) is rejected by this Search service with a 404, while
# the stable 2024-07-01 doesn't recognize newer semantic-config properties
# (e.g. flightingOptIn) that the beta models serialize by default.
# 2024-11-01-preview is the widely-supported preview that accepts both.
_SEARCH_API_VERSION = "2025-05-01-preview"


def _create_search_index_client(options: SearchServiceOptions) -> SearchIndexClient:
    """Create a SearchIndexClient based on available configuration.
    
    Supports two authentication methods:
    1. API Key: Uses AzureKeyCredential with the provided api_key
    2. Managed Identity: Uses DefaultAzureCredential when api_key is not provided
    """
    if options.api_key:
        credential = AzureKeyCredential(options.api_key)
    else:
        credential = DefaultAzureCredential()
    
    return SearchIndexClient(endpoint=options.endpoint, credential=credential, api_version=_SEARCH_API_VERSION)


def _create_search_indexer_client(options: SearchServiceOptions) -> SearchIndexerClient:
    """Create a SearchIndexerClient based on available configuration.
    
    Supports two authentication methods:
    1. API Key: Uses AzureKeyCredential with the provided api_key
    2. Managed Identity: Uses DefaultAzureCredential when api_key is not provided
    """
    if options.api_key:
        credential = AzureKeyCredential(options.api_key)
    else:
        credential = DefaultAzureCredential()
    
    return SearchIndexerClient(endpoint=options.endpoint, credential=credential, api_version=_SEARCH_API_VERSION)


def _create_search_client(options: SearchServiceOptions) -> SearchClient:
    """Create a SearchClient based on available configuration.
    
    Supports two authentication methods:
    1. API Key: Uses AzureKeyCredential with the provided api_key
    2. Managed Identity: Uses DefaultAzureCredential when api_key is not provided
    """
    if options.api_key:
        credential = AzureKeyCredential(options.api_key)
    else:
        credential = DefaultAzureCredential()
    
    return SearchClient(endpoint=options.endpoint, index_name=options.index_name, credential=credential, api_version=_SEARCH_API_VERSION)


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for application services and clients.
    """
    
    config: providers.Singleton[Settings] = providers.Singleton(Settings)
    
    logger: providers.Singleton[Logger] = providers.Singleton(
        create_logger,
        name="knowledge-assistant"
    )

    search_service_options: providers.Singleton[SearchServiceOptions] = providers.Singleton(
        lambda c: c.search_service,
        c=config,
    )

    # Per-language search service options (used by ingestion + chat to pick
    # the right index / data-source / skillset / indexer names).
    search_service_options_en: providers.Singleton[SearchServiceOptions] = providers.Singleton(
        lambda c: c.search_service_options_for("en"),
        c=config,
    )
    search_service_options_fr: providers.Singleton[SearchServiceOptions] = providers.Singleton(
        lambda c: c.search_service_options_for("fr-ca"),
        c=config,
    )

    blob_storage_options: providers.Singleton[BlobStorageOptions] = providers.Singleton(
        lambda c: c.blob_storage,
        c=config,
    )

    blob_storage_options_en: providers.Singleton[BlobStorageOptions] = providers.Singleton(
        lambda c: c.blob_storage_options_for("en"),
        c=config,
    )
    blob_storage_options_fr: providers.Singleton[BlobStorageOptions] = providers.Singleton(
        lambda c: c.blob_storage_options_for("fr-ca"),
        c=config,
    )

    ai_services_options: providers.Singleton[AIServicesOptions] = providers.Singleton(
        lambda c: c.ai_services,
        c=config,
    )

    azure_openai_options: providers.Singleton[AzureOpenAIOptions] = providers.Singleton(
        lambda c: c.azure_openai_options,
        c=config,
    )
    
    azure_ai_foundry_options: providers.Singleton[AzureAIFoundryOptions] = providers.Singleton(
        lambda c: c.azure_ai_foundry_options,
        c=config,
    )

    cosmos_db_options: providers.Singleton[CosmosDBOptions] = providers.Singleton(
        lambda c: c.cosmos_db_options,
        c=config,
    )

    key_vault_options: providers.Singleton[KeyVaultOptions] = providers.Singleton(
        lambda c: c.key_vault_options,
        c=config,
    )

    app_insights_options: providers.Singleton[ApplicationInsightsOptions] = providers.Singleton(
        lambda c: c.app_insights_options,
        c=config,
    )

    workflow_options: providers.Singleton[WorkflowOptions] = providers.Singleton(
        lambda c: c.workflow_options,
        c=config,
    )

    api_options: providers.Singleton[APIOptions] = providers.Singleton(
        lambda c: c.api_options,
        c=config,
    )
    
    search_index_client: providers.Singleton[SearchIndexClient] = providers.Singleton(
        _create_search_index_client,
        options=search_service_options,
    )

    search_indexer_client: providers.Singleton[SearchIndexerClient] = providers.Singleton(
        _create_search_indexer_client,
        options=search_service_options,
    )

    search_client: providers.Singleton[SearchClient] = providers.Singleton(
        _create_search_client,
        options=search_service_options,
    )

    # Per-language Search clients — one bound to each language-specific index.
    search_client_en: providers.Singleton[SearchClient] = providers.Singleton(
        _create_search_client,
        options=search_service_options_en,
    )
    search_client_fr: providers.Singleton[SearchClient] = providers.Singleton(
        _create_search_client,
        options=search_service_options_fr,
    )
    
    cosmos_client: providers.Factory[CosmosClient] = providers.Factory(
        _create_cosmos_client,
        options=cosmos_db_options,
    )
    
    data_source_service: providers.Singleton[IDataSourceService] = providers.Singleton(
        DataSourceService,
        indexer_client=search_indexer_client,
        blob_options=blob_storage_options,
        logger=logger,
    )

    search_index_service: providers.Singleton[ISearchIndexService] = providers.Singleton(
        SearchIndexService,
        index_client=search_index_client,
        openai_options=azure_openai_options,
        logger=logger,
    )

    skillset_service: providers.Singleton[ISkillsetService] = providers.Singleton(
        SkillsetService,
        search_indexer_client=search_indexer_client,
        search_options=search_service_options,
        openai_options=azure_openai_options,
        ai_services_options=ai_services_options,
        blob_options=blob_storage_options,
        logger=logger,
    )

    indexer_service: providers.Singleton[IIndexerService] = providers.Singleton(
        IndexerService,
        indexer_client=search_indexer_client,
        logger=logger,
    )

    search_pipeline_orchestrator: providers.Singleton[ISearchPipelineOrchestrator] = providers.Singleton(
        SearchPipelineOrchestrator,
        data_source_service=data_source_service,
        search_index_service=search_index_service,
        skillset_service=skillset_service,
        indexer_service=indexer_service,
        search_options=search_service_options,
        logger=logger,
        settings=config,
    )
    
    search_service: providers.Factory[ISearchService] = providers.Factory(
        SearchService,
        search_client=search_client,
        openai_options=azure_openai_options,
        logger=logger,
        min_reranker_score=search_service_options.provided.min_reranker_score,
    )

    # Per-language SearchService instances (each bound to a different index
    # and using the matching ``semanticconfig-<lang>`` configuration).
    search_service_en: providers.Singleton[ISearchService] = providers.Singleton(
        SearchService,
        search_client=search_client_en,
        openai_options=azure_openai_options,
        logger=logger,
        index_name=search_service_options_en.provided.index_name,
        min_reranker_score=search_service_options_en.provided.min_reranker_score,
        language="en",
    )
    search_service_fr: providers.Singleton[ISearchService] = providers.Singleton(
        SearchService,
        search_client=search_client_fr,
        openai_options=azure_openai_options,
        logger=logger,
        index_name=search_service_options_fr.provided.index_name,
        min_reranker_score=search_service_options_fr.provided.min_reranker_score,
        language="fr-ca",
    )

    @staticmethod
    def _build_search_service_resolver(en_service, fr_service):
        """Return a callable mapping language code -> SearchService."""
        from app.core.language import Language

        def _resolve(language: str):
            return fr_service if Language.from_value(language) is Language.FRENCH_CANADIAN else en_service

        return _resolve

    search_service_resolver: providers.Singleton = providers.Singleton(
        _build_search_service_resolver,
        en_service=search_service_en,
        fr_service=search_service_fr,
    )
    
    citation_tracker: providers.Factory[CitationTracker] = providers.Factory(
        CitationTracker,
        logger=logger,
    )

    query_rewriter: providers.Factory[QueryRewriter] = providers.Factory(
        QueryRewriter,
        settings=config,
        logger=logger,
    )

    reflection_agent: providers.Factory[ReflectionAgent] = providers.Factory(
        ReflectionAgent,
        settings=config,
        logger=logger,
        workflow_options=workflow_options,
    )

    answer_generator: providers.Factory[AnswerGenerator] = providers.Factory(
        AnswerGenerator,
        settings=config,
        logger=logger,
        citation_tracker=citation_tracker,
    )
    
    agentic_rag_workflow: providers.Factory[AgenticRAGWorkflow] = providers.Factory(
        AgenticRAGWorkflow,
        settings=config,
        logger=logger,
        workflow_options=workflow_options,
        search_service=search_service_resolver,
        citation_tracker=citation_tracker,
        query_rewriter=query_rewriter,
        answer_generator=answer_generator,
        reflection_agent=reflection_agent,
    )
    
    conversation_service: providers.Factory[IConversationService] = providers.Factory(
        ConversationService,
        cosmos_client=cosmos_client,
        database_name=cosmos_db_options.provided.database_name,
        container_name=cosmos_db_options.provided.container_name,
        logger=logger,
    )

    chat_service: providers.Factory[IChatService] = providers.Factory(
        ChatService,
        search_service=search_service,
        openai_options=azure_openai_options,
        logger=logger,
        workflow_options=workflow_options,
        conversation_service=conversation_service,
        workflow=agentic_rag_workflow,
    )