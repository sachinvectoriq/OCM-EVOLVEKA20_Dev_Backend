"""Cosmos persistence for backend-generated API request audit records."""

import asyncio
import json
import logging

from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient, ContainerProxy

logger = logging.getLogger("knowledge-assistant.api-audit")


class APIRequestAuditStore:
    def __init__(self, container: ContainerProxy, write_timeout_seconds: float):
        self.container = container
        self.write_timeout_seconds = write_timeout_seconds

    @classmethod
    async def connect(
        cls,
        client: CosmosClient,
        database_name: str,
        container_name: str,
        *,
        auto_create: bool = False,
        write_timeout_seconds: float = 2.0,
    ) -> "APIRequestAuditStore":
        if auto_create:
            database = await client.create_database_if_not_exists(id=database_name)
            container = await database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/user_id"),
                # Retention is an operational choice, independent of chat history TTL.
                default_ttl=-1,
            )
        else:
            database = client.get_database_client(database_name)
            container = database.get_container_client(container_name)
        properties = await container.read()
        if properties.get("partitionKey", {}).get("paths") != ["/user_id"]:
            raise ValueError("API audit container must use partition key /user_id")
        return cls(container, write_timeout_seconds)

    async def write(self, document: dict) -> None:
        try:
            await asyncio.wait_for(
                self.container.create_item(body=document),
                timeout=self.write_timeout_seconds,
            )
        except Exception as error:
            # Preserve the sanitized record in application logs if Cosmos is down.
            # Never emit exception text: SDK errors can contain request details.
            # This is a fallback, not a durable Cosmos retry queue.
            logger.error(
                "API_AUDIT_WRITE_FAILED error_type=%s record=%s",
                type(error).__name__, json.dumps(document, ensure_ascii=True),
            )
