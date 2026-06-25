
"""
Unified settings loader with Pydantic validation.

Loads configuration from:
1. Environment variables
2. .env file (located in backend/.env)
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.config_options import (
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


class SearchServiceSettings(BaseSettings):
    """Azure AI Search settings for indexing and retrieval."""
    
    model_config = SettingsConfigDict(
        env_prefix='SEARCHSERVICE_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    endpoint: str = Field(..., description="Azure AI Search service endpoint URL")
    api_key: str = Field(..., description="Azure AI Search admin API key")
    skillset_api_version: str = Field(default="2025-08-01-preview", description="Preview API version for skillset")
    # English (default) resource names
    index_name: str = Field(default="knowledge-assistant-content-index", description="Name of the English search index")
    data_source_name: str = Field(default="knowledge-assistant-content-datasource", description="Name of the English data source connection")
    skillset_name: str = Field(default="knowledge-assistant-content-skillset", description="Name of the English skillset")
    indexer_name: str = Field(default="knowledge-assistant-content-indexer", description="Name of the English indexer")
    # French-Canadian resource names (Evolve V2 dual-pipeline)
    index_name_fr: str = Field(default="knowledge-assistant-content-index-fr", description="Name of the French-Canadian search index")
    data_source_name_fr: str = Field(default="knowledge-assistant-content-datasource-fr", description="Name of the French-Canadian data source connection")
    skillset_name_fr: str = Field(default="knowledge-assistant-content-skillset-fr", description="Name of the French-Canadian skillset")
    indexer_name_fr: str = Field(default="knowledge-assistant-content-indexer-fr", description="Name of the French-Canadian indexer")
    min_reranker_score: float = Field(default=2.0, description="Minimum reranker score to retain a result when semantic ranking is enabled (0-4)")


class BlobStorageSettings(BaseSettings):
    """Azure Blob Storage settings for document storage."""
    
    model_config = SettingsConfigDict(
        env_prefix='BLOBSTORAGE_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    resource_id: Optional[str] = Field(None, description="Azure Storage Account Resource ID for managed identity")
    connection_string: Optional[str] = Field(None, description="Connection string for key-based authentication")
    container_name: str = Field(default="documents", description="Blob container name for English source documents")
    container_name_fr: str = Field(default="documents-fr", description="Blob container name for French-Canadian source documents")
    images_container_name: str = Field(default="normalized-images", description="Blob container name for English normalized images")
    images_container_name_fr: str = Field(default="normalized-images-fr", description="Blob container name for French-Canadian normalized images")


class AIServicesSettings(BaseSettings):
    """Azure AI Services settings."""
    
    model_config = SettingsConfigDict(
        env_prefix='AISERVICES_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    cognitive_services_endpoint: str = Field(..., description="Azure AI Services endpoint URL")
    cognitive_services_key: Optional[str] = Field(None, description="API key for Azure AI Services")


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI settings for embeddings and chat completions."""
    
    model_config = SettingsConfigDict(        
        env_prefix='AZURE_OPENAI_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: Optional[str] = Field(None, description="API key for Azure OpenAI")
    deployment_name: str = Field(default="gpt-4", description="GPT-4 deployment name")
    embedding_deployment_name: str = Field(default="text-embedding-3-large", description="Embedding model deployment")
    api_version: str = Field(default="2024-12-01-preview", description="Azure OpenAI API version")
    
    # Separate endpoint for chat completion
    chat_completion_resource_uri: Optional[str] = Field(None, description="Optional separate endpoint for chat completion")
    chat_completion_api_key: Optional[str] = Field(None, description="Optional separate API key for chat completion")
    
    # Model parameters
    temperature: float = Field(default=0.0, description="Temperature for LLM calls")
    max_tokens: int = Field(default=4096, description="Maximum tokens for response")
    max_context_tokens: int = Field(default=128000, description="Maximum context window size")


class CosmosDBSettings(BaseSettings):
    """Azure Cosmos DB settings for conversation history."""
    
    model_config = SettingsConfigDict(
        env_prefix='COSMOS_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    endpoint: Optional[str] = Field(None, description="Cosmos DB endpoint URL (e.g., https://<account>.documents.azure.com:443/) - used with managed identity")
    connection_string: Optional[str] = Field(None, description="Connection string for dev/test only (e.g., AccountEndpoint=...;AccountKey=...;) - not recommended for production")
    database_name: str = Field(default="agentic_rag", description="Database name")
    container_name: str = Field(default="conversations", description="Container name for conversations")
    enable_ttl: bool = Field(default=True, description="Enable time-to-live for conversations")
    default_ttl_days: int = Field(default=30, description="Default TTL in days")


class KeyVaultSettings(BaseSettings):
    """Azure Key Vault settings."""
    
    model_config = SettingsConfigDict(
        env_prefix='KEYVAULT_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    url: Optional[str] = Field(None, description="Key Vault URL")
    use_key_vault: bool = Field(default=False, description="Whether to use Key Vault for secrets")


class ApplicationInsightsSettings(BaseSettings):
    """Application Insights settings."""
    
    model_config = SettingsConfigDict(
        env_prefix='APPINSIGHTS_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    connection_string: Optional[str] = Field(None, description="Application Insights connection string")
    enabled: bool = Field(default=True, description="Enable Application Insights telemetry")


class WorkflowSettings(BaseSettings):
    """Workflow execution settings."""
    
    model_config = SettingsConfigDict(
        env_prefix='WORKFLOW_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    # Execution limits
    max_execution_time: int = Field(default=60, description="Maximum workflow execution time in seconds")
    max_retrieval_iterations: int = Field(default=3, description="Maximum retrieval iterations")
    max_reflection_cycles: int = Field(default=2, description="Maximum reflection cycles")
    conversation_history_window: int = Field(default=5, description="Number of recent conversation messages to include as context for the LLM")
    
    # Query settings
    enable_query_rewriting: bool = Field(default=True, description="Enable query rewriting")
    enable_reflection: bool = Field(default=True, description="Enable reflection loop")
    enable_multi_step_retrieval: bool = Field(default=True, description="Enable multi-step retrieval")

    # Smart-retry thresholds
    reflection_high_validity_threshold: float = Field(default=0.8, description="Valid-result rate above which 'finalize' is overridden to 'retry'")
    reflection_moderate_validity_threshold: float = Field(default=0.6, description="Valid-result rate above which 'finalize' is overridden to 'retry' when combined with a minimum count")
    reflection_moderate_validity_min_count: int = Field(default=3, description="Minimum valid results required for the moderate-validity override")
    
    # Performance settings
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel query execution")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")


class APISettings(BaseSettings):
    """API server settings."""
    
    model_config = SettingsConfigDict(
        env_prefix='API_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per user per minute")
    enable_auth: bool = Field(default=False, description="Enable authentication")
    
    # CORS settings
    enable_cors: bool = Field(default=True, description="Enable CORS (Cross-Origin Resource Sharing)")
    
    # API documentation
    enable_docs: bool = Field(default=True, description="Enable Swagger/OpenAPI documentation endpoints")


class Settings(BaseSettings):
    """
    Unified application settings loaded from environment variables or .env file.
    """

    # Azure Services
    search_service: SearchServiceSettings = Field(default_factory=SearchServiceSettings)
    blob_storage: BlobStorageSettings = Field(default_factory=BlobStorageSettings)
    ai_services: AIServicesSettings = Field(default_factory=AIServicesSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    cosmos_db: CosmosDBSettings = Field(default_factory=CosmosDBSettings)
    key_vault: KeyVaultSettings = Field(default_factory=KeyVaultSettings)
    app_insights: ApplicationInsightsSettings = Field(default_factory=ApplicationInsightsSettings)

    # Application Settings
    environment: str = Field(default="development", description="Environment: development, staging, production")
    use_managed_identity: bool = Field(default=False, description="Use Azure Managed Identity for authentication")
    default_language: str = Field(default="en", description="Default content language when no language is supplied (en | fr-ca)")
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    api: APISettings = Field(default_factory=APISettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
        
    @property
    def search_service_options(self) -> SearchServiceOptions:
        """Create SearchServiceOptions from nested settings (English by default)."""
        return self.search_service_options_for("en")

    def search_service_options_for(self, language: str) -> SearchServiceOptions:
        """Create SearchServiceOptions for a specific language ('en' or 'fr-ca'/'fr')."""
        from app.core.language import Language
        lang = Language.from_value(language)
        if lang is Language.FRENCH_CANADIAN:
            return SearchServiceOptions(
                endpoint=self.search_service.endpoint,
                api_key=self.search_service.api_key,
                skillset_api_version=self.search_service.skillset_api_version,
                index_name=self.search_service.index_name_fr,
                data_source_name=self.search_service.data_source_name_fr,
                skillset_name=self.search_service.skillset_name_fr,
                indexer_name=self.search_service.indexer_name_fr,
                min_reranker_score=self.search_service.min_reranker_score,
            )
        return SearchServiceOptions(
            endpoint=self.search_service.endpoint,
            api_key=self.search_service.api_key,
            skillset_api_version=self.search_service.skillset_api_version,
            index_name=self.search_service.index_name,
            data_source_name=self.search_service.data_source_name,
            skillset_name=self.search_service.skillset_name,
            indexer_name=self.search_service.indexer_name,
            min_reranker_score=self.search_service.min_reranker_score,
        )

    @property
    def blob_storage_options(self) -> BlobStorageOptions:
        """Create BlobStorageOptions from nested settings (English by default)."""
        return self.blob_storage_options_for("en")

    def blob_storage_options_for(self, language: str) -> BlobStorageOptions:
        """Create BlobStorageOptions for a specific language."""
        from app.core.language import Language
        lang = Language.from_value(language)
        if lang is Language.FRENCH_CANADIAN:
            return BlobStorageOptions(
                resource_id=self.blob_storage.resource_id,
                connection_string=self.blob_storage.connection_string,
                container_name=self.blob_storage.container_name_fr,
                images_container_name=self.blob_storage.images_container_name_fr,
            )
        return BlobStorageOptions(
            resource_id=self.blob_storage.resource_id,
            connection_string=self.blob_storage.connection_string,
            container_name=self.blob_storage.container_name,
            images_container_name=self.blob_storage.images_container_name,
        )

    @property
    def ai_services_options(self) -> AIServicesOptions:
        """Create AIServicesOptions from nested settings."""
        return AIServicesOptions(
            cognitive_services_endpoint=self.ai_services.cognitive_services_endpoint,
            cognitive_services_key=self.ai_services.cognitive_services_key,
        )

    @property
    def azure_openai_options(self) -> AzureOpenAIOptions:
        """Create AzureOpenAIOptions from nested settings."""
        return AzureOpenAIOptions(
            resource_uri=self.azure_openai.endpoint,
            api_key=self.azure_openai.api_key,
            text_embedding_model=self.azure_openai.embedding_deployment_name,
            chat_completion_model=self.azure_openai.deployment_name,
            chat_completion_resource_uri=self.azure_openai.chat_completion_resource_uri,
            chat_completion_api_key=self.azure_openai.chat_completion_api_key,
            chat_completion_api_version=self.azure_openai.api_version,
        )
    
    @property
    def azure_ai_foundry_options(self) -> AzureAIFoundryOptions:
        """Create AzureAIFoundryOptions from consolidated OpenAI settings."""
        return AzureAIFoundryOptions(
            endpoint=self.azure_openai.endpoint,
            api_key=self.azure_openai.api_key,
            deployment_name=self.azure_openai.deployment_name,
            embedding_deployment_name=self.azure_openai.embedding_deployment_name,
            api_version=self.azure_openai.api_version,
            temperature=self.azure_openai.temperature,
            max_tokens=self.azure_openai.max_tokens,
            max_context_tokens=self.azure_openai.max_context_tokens,
        )

    @property
    def cosmos_db_options(self) -> CosmosDBOptions:
        """Create CosmosDBOptions from nested settings."""
        return CosmosDBOptions(
            endpoint=self.cosmos_db.endpoint,
            connection_string=self.cosmos_db.connection_string,
            database_name=self.cosmos_db.database_name,
            container_name=self.cosmos_db.container_name,
            enable_ttl=self.cosmos_db.enable_ttl,
            default_ttl_days=self.cosmos_db.default_ttl_days,
        )

    @property
    def key_vault_options(self) -> KeyVaultOptions:
        """Create KeyVaultOptions from nested settings."""
        return KeyVaultOptions(
            url=self.key_vault.url,
            use_key_vault=self.key_vault.use_key_vault,
        )

    @property
    def app_insights_options(self) -> ApplicationInsightsOptions:
        """Create ApplicationInsightsOptions from nested settings."""
        return ApplicationInsightsOptions(
            connection_string=self.app_insights.connection_string,
            enabled=self.app_insights.enabled,
        )

    @property
    def workflow_options(self) -> WorkflowOptions:
        """Create WorkflowOptions from nested settings."""
        return WorkflowOptions(            
            max_retrieval_iterations=self.workflow.max_retrieval_iterations,            
            conversation_history_window=self.workflow.conversation_history_window,
            enable_query_rewriting=self.workflow.enable_query_rewriting,
            enable_reflection=self.workflow.enable_reflection,
            reflection_high_validity_threshold=self.workflow.reflection_high_validity_threshold,
            reflection_moderate_validity_threshold=self.workflow.reflection_moderate_validity_threshold,
            reflection_moderate_validity_min_count=self.workflow.reflection_moderate_validity_min_count,
        )

    @property
    def api_options(self) -> APIOptions:
        """Create APIOptions from nested settings."""
        return APIOptions(
            host=self.api.host,
            port=self.api.port,
            rate_limit_per_minute=self.api.rate_limit_per_minute,
            enable_auth=self.api.enable_auth,
            enable_cors=self.api.enable_cors,
            enable_docs=self.api.enable_docs,
        )


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings