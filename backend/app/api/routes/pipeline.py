"""Ingestion Pipeline management endpoints.

This module provides FastAPI endpoints for managing the multimodal document
ingestion pipeline, including setup, indexing, and status monitoring.

Endpoints:
    POST   /pipeline/setup-pipeline    - Set up complete ingestion infrastructure
    POST   /pipeline/run-indexer       - Run the indexer to process documents
    GET    /pipeline/indexer-status    - Get indexer execution status    

The pipeline includes:
- Blob storage data source with change detection
- Search index with vector search and semantic configuration
- Skillset with multimodal enrichment (text + image processing)
- Indexer for automatic document processing
"""
import csv
from datetime import datetime, timezone
import io
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse

from azure.core.exceptions import AzureError
from azure.search.documents.indexes.models import SearchIndexerStatus

from app.api.schemas.pipeline import PipelineActionRequest
from app.api.dependencies import (
    get_logger,
    get_settings,
    get_indexer_service,
    get_search_pipeline_orchestrator,
)
from app.core.language import Language, all_languages
from app.core.settings import Settings, CosmosDBSettings
from app.ingestion.indexer_service import IIndexerService
from app.ingestion.search_pipeline_orchestrator import ISearchPipelineOrchestrator
from app.utils.warning_logger import _log_indexer_warnings_to_cosmos
from app.utils.url_utils import extract_filename_from_key
from app.utils.warning_logger import _get_cosmos_container


# Initialize router
router = APIRouter(prefix="/pipeline", tags=["Ingestion Pipeline"])


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def success(
    message: str, data: Dict[str, Any] | None = None, status_code: int = 200
) -> JSONResponse:
    """Create a standardized success response."""
    body: Dict[str, Any] = {"message": message}
    if data:
        body["data"] = data
    return JSONResponse(status_code=status_code, content=body)


def error(
    status_code: int, message: str, details: str | None = None
) -> JSONResponse:
    """Create a standardized error response."""
    body = {"error": message}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


# ------------------------------------------------------------------
# Setup pipeline
# ------------------------------------------------------------------


@router.post("/setup-pipeline")
async def setup_pipeline(
    request_body: PipelineActionRequest = Body(default=PipelineActionRequest()),
    orchestrator: ISearchPipelineOrchestrator = Depends(get_search_pipeline_orchestrator),
    indexer_service: IIndexerService = Depends(get_indexer_service),
    settings: Settings = Depends(get_settings),
    logger = Depends(get_logger),
) -> JSONResponse:
    """
    Set up the complete ingestion pipeline infrastructure.

    This endpoint orchestrates the creation of:
    - Blob storage data source with change detection
    - Search index with vector search and semantic configuration
    - Skillset with multimodal enrichment capabilities
    - Indexer to process documents

    Request Body (optional JSON):
    {
        "reset": false  // If true, resets indexer before setup
    }

    Returns:
        JSONResponse with pipeline setup status
    """
    logger.info("Setup pipeline endpoint triggered")

    try:
        reset_indexer: bool = request_body.reset

        # Resolve target languages: explicit override OR every supported language.
        if request_body.language:
            target_langs = [Language.from_value(request_body.language)]
        else:
            target_langs = list(all_languages())

        logger.info(
            f"Starting pipeline setup for languages: {[l.short_code for l in target_langs]}"
        )
        await orchestrator.setup_all_languages_async(
            languages=[l.value for l in target_langs]
        )
        logger.info("Pipeline setup completed successfully")

        # Optionally reset indexers if requested
        reset_indexers: list[str] = []
        if reset_indexer:
            for lang in target_langs:
                opts = settings.search_service_options_for(lang.value)
                logger.info(f"Resetting indexer '{opts.indexer_name}' ({lang.short_code})...")
                await indexer_service.reset_indexer_async(opts.indexer_name)
                reset_indexers.append(opts.indexer_name)
            logger.info("Indexer reset completed")

        return success(
            "Pipeline setup completed successfully",
            {
                "reset": reset_indexer,
                "languages": [l.value for l in target_langs],
                "indexes": [
                    settings.search_service_options_for(l.value).index_name
                    for l in target_langs
                ],
                "reset_indexers": reset_indexers,
            },
        )

    except AzureError as e:
        logger.error(f"Azure error during pipeline setup: {str(e)}")
        return error(500, "Azure service error", str(e))
    except Exception as e:
        logger.error(f"Unexpected error during pipeline setup: {str(e)}")
        return error(500, "Internal server error", str(e))


# ------------------------------------------------------------------
# Run indexer
# ------------------------------------------------------------------


@router.post("/run-indexer")
async def run_indexer(
    request_body: PipelineActionRequest = Body(default=PipelineActionRequest()),
    indexer_service: IIndexerService = Depends(get_indexer_service),
    settings: Settings = Depends(get_settings),
    logger = Depends(get_logger),
) -> JSONResponse:
    """
    Run the search indexer to process documents.

    The indexer processes all documents in the configured blob storage
    container and updates the search index with enriched content.

    Request Body (optional JSON):
    {
        "reset": false  // If true, resets indexer before running
    }

    Returns:
        JSONResponse with indexer run status
    """
    logger.info("Run indexer endpoint triggered")

    try:
        reset_first: bool = request_body.reset

        if request_body.language:
            target_langs = [Language.from_value(request_body.language)]
        else:
            target_langs = list(all_languages())

        triggered: list[str] = []
        for lang in target_langs:
            indexer_name = settings.search_service_options_for(lang.value).indexer_name
            if reset_first:
                logger.info(f"Resetting indexer '{indexer_name}' ({lang.short_code})")
                await indexer_service.reset_indexer_async(indexer_name)
                logger.info("Indexer reset completed")

            logger.info(f"Running indexer '{indexer_name}' ({lang.short_code})")
            await indexer_service.run_indexer_async(indexer_name)
            triggered.append(indexer_name)
        logger.info("Indexer run initiated successfully")

        return success(
            "Indexer run initiated successfully",
            {
                "indexers": triggered,
                "reset": reset_first,
                "languages": [l.value for l in target_langs],
            },
        )

    except AzureError as e:
        logger.error(f"Azure error during indexer run: {str(e)}")
        return error(500, "Azure service error", str(e))
    except Exception as e:
        logger.error(f"Unexpected error during indexer run: {str(e)}")
        return error(500, "Internal server error", str(e))


# ------------------------------------------------------------------
# Indexer status
# ------------------------------------------------------------------


@router.get("/indexer-status")
async def get_indexer_status(
    indexer_name: str | None = None,
    indexer_service: IIndexerService = Depends(get_indexer_service),
    settings: Settings = Depends(get_settings),
    logger = Depends(get_logger),
) -> JSONResponse:
    """
    Get the current status of the indexer.

    Query Parameters:
    - indexer_name (optional): Name of the indexer (defaults to configured name)

    Returns:
        JSONResponse with indexer status information including:
        - Current status (running, idle, error, etc.)
        - Last execution results
        - Error/warning details if any
        - Item counts (processed, failed)
    """
    logger.info("Get indexer status endpoint triggered")

    try:
        # Get indexer name from query params or use default
        name: str = indexer_name or settings.search_service.indexer_name

        # Get indexer status
        logger.info(f"Retrieving status for indexer: {name}")
        status: SearchIndexerStatus = (
            await indexer_service.get_indexer_status_async(name)
        )
#------------------Warning code----------
        warnings_list: List[Dict[str, Optional[str]]] = []
        last_result = getattr(status, "last_result", None)

        if last_result:
            raw_warnings = getattr(last_result, "warnings", None) or []
            warnings_list = [
                {
                    "message": getattr(w, "message", None),
                    "key": getattr(w, "key", None),
                    "file_name": extract_filename_from_key(getattr(w, "key", None)),
                    "name": getattr(w, "name", None),
                    "details": getattr(w, "details", None),
                }
                for w in raw_warnings
            ]

        # Store warnings to Cosmos only if warnings exist
        if warnings_list:
            try:
                _log_indexer_warnings_to_cosmos(settings=settings, indexer_name=name, warnings=warnings_list)
                logger.info(f"Stored {len(warnings_list)} warnings in CosmosDB for indexer: {name}")
            except Exception as log_ex:
                # Do NOT fail endpoint if Cosmos logging fails
                logger.error(f"Failed to store warnings in CosmosDB: {log_ex}")
#------------------------
        # Extract relevant status information
        status_info: Dict[str, Any] = {
            "indexer_name": status.name,
            "status": status.status,
            "last_result": (
                {
                    "status": status.last_result.status,
                    "error_message": getattr(
                        status.last_result, "error_message", None
                    ),
                    "start_time": (
                        status.last_result.start_time.isoformat()
                        if status.last_result.start_time
                        else None
                    ),
                    "end_time": (
                        status.last_result.end_time.isoformat()
                        if status.last_result.end_time
                        else None
                    ),
                    "items_processed": status.last_result.item_count,
                    "items_failed": status.last_result.failed_item_count,
                }
                if status.last_result
                else None
            ),
            "warnings": warnings_list,
        }

        logger.info(f"Indexer status retrieved: {status.status}")
        return success(
            "Indexer status retrieved successfully",
            status_info,
        )

    except AzureError as e:
        logger.error(f"Azure error retrieving indexer status: {str(e)}")
        return error(500, "Azure service error", str(e))
    except Exception as e:
        logger.error(f"Unexpected error retrieving indexer status: {str(e)}")
        return error(500, "Internal server error", str(e))

#------------------Export warnings to CSV endpoint--------------------------  
@router.get("/indexer-warnings/export")
def export_indexer_warnings_csv(
    indexer_name: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    settings: CosmosDBSettings = Depends(get_settings),
):
    """
    Export stored indexer warnings to CSV.

    Reads warnings from Cosmos DB logs (service='search-indexer').
    Returns a downloadable CSV (no disk writes).
    """
    # settings: Settings = container.config()
    cosmos_container = _get_cosmos_container(settings)

    query = """
    SELECT TOP @limit
    c.timestamp AS timestamp,
    c.indexerName AS indexerName,
    c.details.warnings AS warnings
    FROM c
    WHERE c.service = "search-indexer"
    AND IS_DEFINED(c.details.warnings)
    AND ARRAY_LENGTH(c.details.warnings) > 0
    ORDER BY c.timestamp DESC
    """
    params = [{"name": "@limit", "value": limit}]

    items = list(
        cosmos_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        )
    )

    rows: List[Dict[str, Any]] = []
    for item in items:
        ts = item.get("timestamp")
        idx = item.get("indexerName")

        if indexer_name and idx != indexer_name:
            continue

        warnings = item.get("warnings") or []
        for w in warnings:
            if not isinstance(w, dict):
                continue

            rows.append(
                {
                    "timestamp": ts,
                    "indexerName": idx,
                    "warning_name": w.get("name"),
                    "warning_message": w.get("message"),
                    "file_name": w.get("file_name"),
                }
            )

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["timestamp", "indexerName", "warning_name", "warning_message", "file_name"],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)

    filename = f"indexer_warnings_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
