"""
Audit Logging API

This module provides a standalone audit logging endpoint.
Frontend calls this API directly to store audit logs in Cosmos DB.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from azure.cosmos.aio import CosmosClient

from app.core.container import Container

router = APIRouter(prefix="/audit", tags=["Audit"])


# ---------------------------
# Request Model
# ---------------------------

class AuditRequest(BaseModel):
    chat_session_id: str
    user_id: str
    user_name: str
    job_title: str
    query_language: str
    opco: str
    persona: str
    query: str
    ai_response: str
    citations: list


# ---------------------------
# Dependency to Get Cosmos
# ---------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()


# ---------------------------
# API Endpoint
# ---------------------------

@router.post("/log")
async def create_audit_log(
    request: AuditRequest,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        database_name = "audit-evolve"
        container_name = "audit-evolve-container"

        db = cosmos_client.get_database_client(database_name)
        container = db.get_container_client(container_name)

        current_utc = datetime.now(timezone.utc)

        document = {
            "id": str(uuid.uuid4()),
            "chat_session_id": request.chat_session_id,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "job_title": request.job_title,
            "query_language": request.query_language,
            "opco": request.opco,
            "persona": request.persona,
            "timestamp_utc": current_utc.isoformat(),  # full timestamp
            "date": current_utc.date().isoformat(),    # only date (YYYY-MM-DD)
            "query": request.query,
            "ai_response": request.ai_response,
            "citations": request.citations,
        }

        await container.create_item(document)

        return {"message": "Audit log stored successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
