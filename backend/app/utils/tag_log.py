"""
Tag Logs API

This module provides a standalone tag logging endpoint.
Frontend/backend can call this API directly to store tag logs in Cosmos DB.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from azure.cosmos.aio import CosmosClient

from app.core.container import Container


router = APIRouter(prefix="/tag-logs", tags=["Tag Logs"])


# ---------------------------
# Request Model
# ---------------------------

class TagLogRequest(BaseModel):
    doc_id: str
    user_id: str | None = None
    event_type: str | None = None
    source: str | None = None
    metadata: dict | None = None


# ---------------------------
# Dependency to Get Cosmos
# ---------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()


# ---------------------------
# API Endpoint
# ---------------------------

@router.post("/submit")
async def submit_tag_log(
    request: TagLogRequest,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        database_name = "tag_logs"
        container_name = "tag_logs_container"

        db = cosmos_client.get_database_client(database_name)
        container = db.get_container_client(container_name)

        current_utc = datetime.now(timezone.utc)

        document = {
            "id": str(uuid.uuid4()),
            "doc_id": request.doc_id,   # 👈 partition key
            "user_id": request.user_id,
            "event_type": request.event_type,
            "source": request.source,
            "metadata": request.metadata or {},
            "timestamp_utc": current_utc.isoformat(),
            "date": current_utc.date().isoformat(),
            "record_type": "tag_log"
        }

        await container.create_item(document)

        return {"message": "Tag log submitted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))