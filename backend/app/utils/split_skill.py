from fastapi import APIRouter
from typing import Dict, Any, List

# -----------------------------------
# Router (clean naming for WebApiSkill)
# -----------------------------------
split_skill_router = APIRouter()


# -----------------------------------
# Utility: safe comma splitter
# handles: "OSG,FSG", "OSG, FSG , ABC"
# -----------------------------------
def safe_split(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


# -----------------------------------
# Azure WebApiSkill endpoint
# -----------------------------------
@split_skill_router.post("/split")
async def split_skill(payload: Dict[str, Any]):
    """
    Azure expects:
    {
      "values": [
        {
          "recordId": "1",
          "data": {
            "opco": "OSG,FSG",
            "persona": "FSG,OSG"
          }
        }
      ]
    }
    """

    results = []

    for item in payload.get("values", []):
        record_id = item.get("recordId")
        data = item.get("data", {})

        opco_raw = data.get("opco", "")
        persona_raw = data.get("persona", "")

        results.append({
            "recordId": record_id,
            "data": {
                "opco_values": safe_split(opco_raw),
                "persona_values": safe_split(persona_raw)
            }
        })

    return {"values": results}