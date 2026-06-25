"""Data source service for Azure AI Search."""

from abc import ABC, abstractmethod
from typing import Optional

from azure.search.documents.indexes.aio import SearchIndexerClient
from azure.search.documents.indexes.models import (
    HighWaterMarkChangeDetectionPolicy,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataSourceType,
    SoftDeleteColumnDeletionDetectionPolicy,
)

from app.models.config_options import BlobStorageOptions


class IDataSourceService(ABC):
    """Interface for data source service operations."""

    @abstractmethod
    async def create_blob_data_source_async(
        self,
        data_source_name: str,
        blob_options: Optional[BlobStorageOptions] = None,
    ) -> None:
        """Create or update a blob data source connection."""
        pass


class DataSourceService(IDataSourceService):
    """Service for managing Azure AI Search data source connections."""

    def __init__(
        self,
        indexer_client: SearchIndexerClient,
        blob_options: BlobStorageOptions,
        logger,
    ) -> None:
        self._indexer_client: SearchIndexerClient = indexer_client
        self._blob_options: BlobStorageOptions = blob_options
        self.logger = logger

    async def create_blob_data_source_async(
        self,
        data_source_name: str,
        blob_options: Optional[BlobStorageOptions] = None,
    ) -> None:
        """Create or update a blob data source connection.

        ``blob_options`` overrides the constructor-provided options so callers
        can target a language-specific container (e.g. ``documents-fr``).
        """
        opts = blob_options or self._blob_options
        container: SearchIndexerDataContainer = SearchIndexerDataContainer(
            name=opts.container_name
        )

        connection_string: str
        if opts.resource_id:
            connection_string = f"ResourceId={opts.resource_id};"
            self.logger.info("Using managed identity authentication for blob data source")
        elif opts.connection_string:
            connection_string = opts.connection_string
            self.logger.info("Using connection string authentication for blob data source")
        else:
            raise ValueError(
                "Either BlobStorage__ResourceId or BlobStorageConnection must be configured"
            )

        data_source: SearchIndexerDataSourceConnection = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type=SearchIndexerDataSourceType.AZURE_BLOB,
            connection_string=connection_string,
            container=container,
            description="A data source to store multi-modality documents",
            data_change_detection_policy=HighWaterMarkChangeDetectionPolicy(
                high_water_mark_column_name="metadata_storage_last_modified"
            ),
            data_deletion_detection_policy=SoftDeleteColumnDeletionDetectionPolicy(
                soft_delete_column_name="metadata_storage_is_deleted",
                soft_delete_marker_value="true",
            ),
        )

        await self._indexer_client.create_or_update_data_source_connection(data_source)
        self.logger.info(
            f"Data source '{data_source_name}' (container='{opts.container_name}') created or updated."
        )
