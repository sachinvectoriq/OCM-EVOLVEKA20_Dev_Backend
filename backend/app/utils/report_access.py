"""
Report Access API

Handles inserting, fetching, and deleting
report access logs from Cosmos DB.
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
# Request Models
# ---------------------------

class ReportAccessRequest(BaseModel):
    user_mail: str
    user_name: str
    provider_name: str


class DeleteReportAccessRequest(BaseModel):
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
        db = cosmos_client.get_database_client("report-access-table-evolve")
        container = db.get_container_client("report-access-table-evolve")

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
        db = cosmos_client.get_database_client("report-access-table-evolve")
        container = db.get_container_client("report-access-table-evolve")

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
# 3️⃣ Delete Record
# ---------------------------

@router.delete("/delete")
async def delete_report_access(
    request: DeleteReportAccessRequest,
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):
    try:
        db = cosmos_client.get_database_client("report-access-table-evolve")
        container = db.get_container_client("report-access-table-evolve")

        query = """
        SELECT * FROM c
        WHERE c.user_mail = @user_mail
        AND c.user_name = @user_name
        AND c.record_type = 'report_access'
        """

        parameters = [
            {"name": "@user_mail", "value": request.user_mail},
            {"name": "@user_name", "value": request.user_name},
        ]

        items = []

        async for item in container.query_items(
            query=query,
            parameters=parameters
        ):
            items.append(item)

        if not items:
            raise HTTPException(
                status_code=404,
                detail="No matching record found"
            )

        for item in items:
            await container.delete_item(
                item=item["id"],
                partition_key=item["user_mail"]
            )

        return {
            "message": "Report access record(s) deleted successfully",
            "deleted_count": len(items)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
