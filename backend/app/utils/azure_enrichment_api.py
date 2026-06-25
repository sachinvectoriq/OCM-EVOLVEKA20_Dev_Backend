from fastapi import APIRouter
from typing import Dict, List, Any
import re

router = APIRouter()

# ================================
# HELPERS
# ================================
_SPLIT_PATTERN = re.compile(r"[|,;]")


def parse_values(raw: Any) -> List[str]:
    """
    Converts:
      'fsg'           -> ['fsg']
      'fsg,tsg'       -> ['fsg', 'tsg']
      'fsg, tsg'      -> ['fsg', 'tsg']
      'fsg|tsg'       -> ['fsg', 'tsg']
      None / ''       -> []
    """
    if raw is None:
        return []
    return [x.strip() for x in _SPLIT_PATTERN.split(str(raw)) if x and x.strip()]


# ================================
# MAIN WEB API SKILL ENDPOINT
# ================================
@router.post("/enrich")
async def enrich(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Azure WebApiSkill entry point.
    Expects each record's `data` to contain `opco` and `persona`
    (sourced from /document/opco and /document/persona in the skillset).
    """
    results = []

    for item in request.get("values", []):
        data = item.get("data", {}) or {}

        results.append({
            "recordId": item.get("recordId"),
            "data": {
                "opco_values":    parse_values(data.get("opco")),
                "persona_values": parse_values(data.get("persona")),
            },
        })

    return {"values": results}
