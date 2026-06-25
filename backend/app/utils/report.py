from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from azure.cosmos.aio import CosmosClient
from app.core.container import Container
from datetime import datetime

router = APIRouter(
    prefix="/audit-report",
    tags=["Audit Report"]
)

# ---------------------------------------------------
# Dependency
# ---------------------------------------------------

def get_cosmos_client():
    container = Container()
    return container.cosmos_client()

# ---------------------------------------------------
# Excluded Users
# ---------------------------------------------------

EXCLUDED_USERS = [
    "Bhaskar, Solomon",
    "Sachin Bhusanurmath",
    "Jain, Anshuman",
    "HardCodedUser",
    "Anonymous",
    "Test User",
    "string",
    "Chanbasava Koti",
    "User1",
    "Bhusanurmath, Sachin",
    "Sai Charan Kumbham",
    "User",
    "Tester",
    "sachin-test"
]

# ---------------------------------------------------
# Master OPCO / Persona Values (UPDATED ONLY THIS PART)
# ---------------------------------------------------

OPCOS = [
    "Actalent",
    "Actalent Services",
    "Aerotek",
    "Aerotek Services",
    "Aston Carter",
    "TEKsystems",
    "TEKsystems Global Services",
    "Allegis Corporate Services",
    "TEK/TGS NA",
    "ACS"
]

PERSONAS = [
    "FSG",
    "CLS",
    "Sales and Recruiting",
    "Delivery and TA Services",
    "Front Office",
    "Back Office",
    "Corporate Services",
    "Talent",
    "TEK Talent Delivery/MSP Lead/OM/EM",
    "TGS Recruiter",
    "TGS Delivery",
    "TEK Sales/MSP Directors",
    "TGS Sales",
    "Accounting Operations",
    "Corporate",
    "Field Support Group",
    "Operational Risk & Compliance",
    "External Users",
    "Employee Self-Service",
    "Supervisor/Manager/Leader Self-Service"
]

# ---------------------------------------------------
# Language Mapping
# ---------------------------------------------------

LANGUAGE_MAP = {
    "en": "English",
    "fr": "French"
}

def format_language(value: Optional[str]) -> str:
    if not value:
        return "-"
    return LANGUAGE_MAP.get(value.strip().lower(), value)

# ---------------------------------------------------
# Create Normalized Lookup Maps
# ---------------------------------------------------

def normalize_value(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "")
        .replace("-", "")
    )

OPCO_LOOKUP = {
    normalize_value(opco): opco
    for opco in OPCOS
}

PERSONA_LOOKUP = {
    normalize_value(persona): persona
    for persona in PERSONAS
}

# ---------------------------------------------------
# Post Processing Formatter
# ---------------------------------------------------

def format_opco(value: Optional[str]) -> str:
    if not value:
        return "-"
    normalized = normalize_value(value)
    return OPCO_LOOKUP.get(normalized, value)

def format_persona(value: Optional[str]) -> str:
    if not value:
        return "-"
    normalized = normalize_value(value)
    return PERSONA_LOOKUP.get(normalized, value)

# ---------------------------------------------------
# Combined Audit + Feedback Report
# ---------------------------------------------------

@router.get("/combined-report")
async def combined_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_name: Optional[str] = None,
    persona: Optional[str] = None,
    opco: Optional[str] = None,
    feedback_type: Optional[str] = None,
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):

    try:
        audit_db = cosmos_client.get_database_client("audit-evolve")
        audit_container = audit_db.get_container_client("audit-evolve-container")

        feedback_db = cosmos_client.get_database_client("feedback-evolve")
        feedback_container = feedback_db.get_container_client("feedback-evolve-container")

        conditions = []
        parameters = []

        excluded_users_query = ",".join([f"'{user}'" for user in EXCLUDED_USERS])
        conditions.append(f"c.user_name NOT IN ({excluded_users_query})")

        if user_name:
            conditions.append("c.user_name = @user_name")
            parameters.append({"name": "@user_name", "value": user_name})

        if persona:
            normalized_persona = normalize_value(persona)
            conditions.append(
                "REPLACE(REPLACE(LOWER(c.persona), ' ', ''), '-', '') = @persona"
            )
            parameters.append({"name": "@persona", "value": normalized_persona})

        if opco:
            normalized_opco = normalize_value(opco)
            conditions.append(
                "REPLACE(REPLACE(LOWER(c.opco), ' ', ''), '-', '') = @opco"
            )
            parameters.append({"name": "@opco", "value": normalized_opco})

        if start_date:
            conditions.append("c.date >= @start_date")
            parameters.append({"name": "@start_date", "value": start_date})

        if end_date:
            conditions.append("c.date <= @end_date")
            parameters.append({"name": "@end_date", "value": end_date})

        where_clause = " AND ".join(conditions)

        audit_query = f"""
        SELECT
            c.id,
            c.chat_session_id,
            c.user_id,
            c.user_name,
            c.job_title,
            c.opco,
            c.persona,
            c.query_language,
            c.timestamp_utc,
            c.date,
            c.query,
            c.ai_response,
            c.citations
        FROM c
        WHERE {where_clause}
        ORDER BY c._ts DESC
        OFFSET {offset} LIMIT {limit}
        """

        audit_items = []
        async for item in audit_container.query_items(
            query=audit_query,
            parameters=parameters
        ):
            audit_items.append(item)

        feedback_query = """
        SELECT
            c.chat_session_id,
            c.query,
            c.ai_response,
            c.feedback_type,
            c.feedback_text
        FROM c
        WHERE c.record_type = 'feedback'
        """

        feedback_items = []
        async for item in feedback_container.query_items(query=feedback_query):
            feedback_items.append(item)

        feedback_lookup = {}

        for fb in feedback_items:
            key = (
                fb.get("chat_session_id"),
                fb.get("query"),
                fb.get("ai_response")
            )

            feedback_lookup[key] = {
                "feedback_type": fb.get("feedback_type", "").strip().lower(),
                "feedback_note": fb.get("feedback_text", "-")
            }

        final_results = []

        for audit in audit_items:

            key = (
                audit.get("chat_session_id"),
                audit.get("query"),
                audit.get("ai_response")
            )

            feedback_data = feedback_lookup.get(key)

            if feedback_type:
                requested_feedback = feedback_type.strip().lower()

                if not feedback_data:
                    continue

                actual_feedback = feedback_data.get("feedback_type", "").strip().lower()

                if actual_feedback != requested_feedback:
                    continue

            if not feedback_data:
                feedback_data = {
                    "feedback_type": "-",
                    "feedback_note": "-"
                }

            formatted_timestamp = audit.get("timestamp_utc", "-")

            try:
                if formatted_timestamp:
                    parsed_time = datetime.fromisoformat(
                        formatted_timestamp.replace("Z", "+00:00")
                    )
                    formatted_timestamp = parsed_time.strftime("%b %d, %Y, %I:%M %p")
            except Exception:
                pass

            combined_row = {
                "user_name": audit.get("user_name", "-"),
                "job_title": audit.get("job_title", "-"),
                "opco": format_opco(audit.get("opco")),
                "persona": format_persona(audit.get("persona")),
                "query_language": format_language(audit.get("query_language")),
                "query": audit.get("query", "-"),
                "ai_response": audit.get("ai_response", "-"),
                "citations": audit.get("citations", "-"),
                "date_and_time": formatted_timestamp,
                "feedback_type": feedback_data["feedback_type"],
                "feedback_note": feedback_data["feedback_note"]
            }

            final_results.append(combined_row)

        return {
            "count": len(final_results),
            "limit": limit,
            "offset": offset,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "user_name": user_name,
                "persona": persona,
                "opco": opco,
                "feedback_type": feedback_type
            },
            "data": final_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# Get Unique User Names
# ---------------------------------------------------

@router.get("/users")
async def get_unique_users(
    cosmos_client: CosmosClient = Depends(get_cosmos_client),
):

    try:
        audit_db = cosmos_client.get_database_client("audit-evolve")
        audit_container = audit_db.get_container_client("audit-evolve-container")

        query = """
        SELECT DISTINCT c.user_name
        FROM c
        WHERE IS_DEFINED(c.user_name)
        """

        users = set()

        async for item in audit_container.query_items(query=query):
            user_name = item.get("user_name")

            if user_name and user_name not in EXCLUDED_USERS:
                users.add(user_name)

        final_users = sorted(list(users))

        return {
            "count": len(final_users),
            "users": final_users
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
