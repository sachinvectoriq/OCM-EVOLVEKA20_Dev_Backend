from __future__ import annotations
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from app.core.settings import Settings, CosmosDBSettings

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid

from azure.cosmos import CosmosClient
from azure.core.exceptions import AzureError

_cosmos_client = None
_cosmos_container_cache = {}  # cache per db/container


def _get_cosmos_container(
    settings,
    database_name: str = "warning-table",
    container_name: str = "warning-container",
):
    """
    Lazily initialize Cosmos client and return container client.
    Uses manual database/container names passed from caller.
    Cache key: (database_name, container_name)
    """
    global _cosmos_client, _cosmos_container_cache

    cache_key = (database_name, container_name)
    if cache_key in _cosmos_container_cache:
        return _cosmos_container_cache[cache_key]

    cosmos = getattr(settings, "cosmos_db", None)
    if cosmos is None:
        raise ValueError("Settings missing 'cosmos_db' (CosmosDBSettings).")

    # Create client once
    if _cosmos_client is None:
        if getattr(cosmos, "connection_string", None):
            _cosmos_client = CosmosClient.from_connection_string(cosmos.connection_string)
        elif getattr(cosmos, "endpoint", None):
            _cosmos_client = CosmosClient(url=cosmos.endpoint, credential=DefaultAzureCredential())
        else:
            raise ValueError(
                "Cosmos config missing. Provide COSMOS_CONNECTION_STRING (dev/test) or COSMOS_ENDPOINT (managed identity)."
            )

    db = _cosmos_client.get_database_client(database_name)
    container = db.get_container_client(container_name)

    _cosmos_container_cache[cache_key] = container
    return container


def _log_indexer_warnings_to_cosmos(
    settings: Settings,
    indexer_name: str,
    warnings: List[Dict[str, Optional[str]]],
) -> None:
    """
    Store warnings in Cosmos DB using your log_item schema.
    Store ONLY when warnings exist.
    """
    if not warnings:
        return

    container = _get_cosmos_container(settings)

    message_text = f"Indexer warnings detected ({len(warnings)})."

    log_item: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "service": "search-indexer",  # partition key (as per your schema)
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "WARN",
        "no_of_warnings": message_text,
        "indexerName": indexer_name,
        "details": {
            "warnings": warnings,
        },
    }

    try:
        container.create_item(body=log_item)
    except AzureError as e:
        # If Cosmos fails, we raise so caller can decide to swallow it
        raise e