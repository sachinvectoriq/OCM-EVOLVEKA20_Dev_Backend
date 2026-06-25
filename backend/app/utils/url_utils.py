import re
from typing import Optional
from urllib.parse import unquote


def extract_filename_from_key(key: Optional[str]) -> Optional[str]:
    """
    Extract filename from an indexer warning key (URL-encoded blob URL or path).
    Returns None if key is empty or cannot be parsed.
    """
    if not key or not isinstance(key, str):
        return None

    try:
        decoded = unquote(key)
        decoded = decoded.split("?", 1)[0]
        un_filename = decoded.rsplit("/", 1)[-1].strip()
        filename = re.sub(r"%20", " ", un_filename)
        return filename or None
    except Exception:
        return None