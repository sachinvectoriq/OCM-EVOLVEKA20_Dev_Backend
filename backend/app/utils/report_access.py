"""
Report Access API

Handles inserting and fetching report access logs
from Cosmos DB.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from azure.cosmos.aio import CosmosClient

from app.core.container import Container


# No tags used here
router = APIRouter(prefix="/report-access")


# ---------------------------
# Request Model
# ---------------------------

class ReportAccessRequest(BaseModel):
    user_mail: str
    user_name: str
    provider_name: str


class ReportAccessDeleteRequest(BaseModel):
    user_mail: str
    user_name: str


# ---------------------------
# Dependency
# ---------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()


# ---------------------------
# 1️⃣ Insert Access Record
# ---------------------------

@router.post("/insert")
async def insert_report_access(
    request: ReportAccessRequest,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        db = cosmos_client.get_database_client("report-access-table")
        container = db.get_container_client("report-access-container")

        document = {
            "id": str(uuid.uuid4()),
            "user_mail": request.user_mail,  # Partition key
            "user_name": request.user_name,
            "provider_name": request.provider_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "record_type": "report_access"
        }

        await container.create_item(document)

        return {"message": "Report access record inserted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# 2️⃣ Fetch All Records
# ---------------------------

@router.get("/all")
async def get_all_report_access(
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        db = cosmos_client.get_database_client("report-access-table")
        container = db.get_container_client("report-access-container")

        query = "SELECT * FROM c WHERE c.record_type = 'report_access'"

        items = []
        async for item in container.query_items(query=query):
            items.append(item)

        return {
            "count": len(items),
            "data": items
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# 3️⃣ Delete Access Record
# ---------------------------

@router.delete("/delete")
async def delete_report_access(
    request: ReportAccessDeleteRequest,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        db = cosmos_client.get_database_client("report-access-table")
        container = db.get_container_client("report-access-container")

        # We don't know the id up front, so look up all matching records by
        # user_mail (the partition key) + user_name, then delete each one.
        query = (
            "SELECT c.id FROM c "
            "WHERE c.record_type = 'report_access' "
            "AND c.user_mail = @user_mail AND c.user_name = @user_name"
        )
        parameters = [
            {"name": "@user_mail", "value": request.user_mail},
            {"name": "@user_name", "value": request.user_name},
        ]

        ids = []
        async for item in container.query_items(
            query=query,
            parameters=parameters,
            partition_key=request.user_mail,
        ):
            ids.append(item["id"])

        if not ids:
            raise HTTPException(status_code=404, detail="Report access record not found")

        # user_mail is the partition key, so it must be supplied for each delete.
        for record_id in ids:
            await container.delete_item(item=record_id, partition_key=request.user_mail)

        return {
            "message": "Report access record(s) deleted successfully",
            "deleted_count": len(ids),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
