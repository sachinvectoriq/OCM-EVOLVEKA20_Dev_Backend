"""
Blob URL signer.

Wraps an Azure Blob Storage URL with a short-lived, read-only SAS token so that
a browser / frontend can open it directly even when the underlying container is
private.

Auth strategy (in priority order):
  1. Managed Identity / DefaultAzureCredential  -> User Delegation SAS
     (preferred, no account keys needed; works when USE_MANAGED_IDENTITY=true)
  2. Connection string with AccountKey          -> Account Key SAS
     (development fallback only)

The user delegation key is cached per storage account and refreshed before
expiry. SAS lifetime defaults to 60 minutes which is plenty for a user to
click a citation.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    UserDelegationKey,
    generate_blob_sas,
)


_BLOB_HOST_SUFFIX = ".blob.core.windows.net"
_DEFAULT_TTL_MINUTES = 60
_UDK_REQUEST_HOURS = 6      # how long Azure should make the delegation key valid
_UDK_REFRESH_SKEW = timedelta(minutes=5)


# Extension -> (Content-Type, prefer-inline-via-Office-viewer)
# These map common document types to the MIME types that Office Online and
# browsers expect. Anything not listed falls back to no override and the
# blob's stored Content-Type is used as-is.
_EXT_CONTENT_TYPE = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".txt": "text/plain; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
}


class BlobUrlSigner:
    """Generate read-only SAS-signed URLs for Azure blob URLs."""

    def __init__(
        self,
        use_managed_identity: bool,
        connection_string: Optional[str] = None,
        ttl_minutes: int = _DEFAULT_TTL_MINUTES,
    ) -> None:
        self._use_mi = use_managed_identity
        self._connection_string = connection_string
        self._ttl = timedelta(minutes=ttl_minutes)
        self._credential: Optional[DefaultAzureCredential] = (
            DefaultAzureCredential() if use_managed_identity else None
        )
        self._lock = threading.Lock()
        # account_name -> (UserDelegationKey, expires_at_utc)
        self._udk_cache: dict[str, Tuple[UserDelegationKey, datetime]] = {}

    # ------------------------------------------------------------------ public
    def sign(self, blob_url: str) -> str:
        """Return ``blob_url`` with a SAS appended. Returns input unchanged on
        any failure or when the URL is not an Azure blob URL.

        The SAS also includes response-header overrides (``rsct`` and ``rscd``)
        so the blob is always served with the correct ``Content-Type`` and an
        ``inline`` ``Content-Disposition`` — required for Office Online /
        Microsoft Preview and browser PDF viewers to render the file instead
        of forcing a download.
        """
        if not blob_url or _BLOB_HOST_SUFFIX not in blob_url:
            return blob_url

        split = urlsplit(blob_url)
        if "sig=" in (split.query or ""):
            return blob_url  # already signed

        account_name, container, blob_name = self._parse(split)
        if not (account_name and container and blob_name):
            return blob_url

        expiry = datetime.now(timezone.utc) + self._ttl
        permissions = BlobSasPermissions(read=True)

        content_type, content_disposition = self._response_headers_for(blob_name)

        sas_kwargs: dict = {
            "account_name": account_name,
            "container_name": container,
            "blob_name": blob_name,
            "permission": permissions,
            "expiry": expiry,
        }
        if content_type:
            sas_kwargs["content_type"] = content_type
        if content_disposition:
            sas_kwargs["content_disposition"] = content_disposition

        if self._use_mi:
            sas_kwargs["user_delegation_key"] = self._get_user_delegation_key(
                account_name
            )
            sas = generate_blob_sas(**sas_kwargs)
        else:
            account_key = self._account_key_from_connection_string()
            if not account_key:
                return blob_url
            sas_kwargs["account_key"] = account_key
            sas = generate_blob_sas(**sas_kwargs)

        new_query = (split.query + "&" if split.query else "") + sas

        # Ensure the path is percent-encoded (spaces -> %20, etc.) so the URL
        # is safe to embed directly in JSON / hrefs / Office Viewer src params
        # without relying on the consumer to re-encode it. We decode first so
        # we don't double-encode an already-encoded path.
        encoded_path = quote(unquote(split.path), safe="/")

        return urlunsplit(
            (split.scheme, split.netloc, encoded_path, new_query, split.fragment)
        )

    @staticmethod
    def _response_headers_for(blob_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Decide what Content-Type and Content-Disposition to force via SAS.

        Returns (content_type, content_disposition). Either may be None to
        leave the stored value untouched.
        """
        # The blob_name in the URL path is percent-encoded; the SAS overrides
        # need the *raw* filename for Content-Disposition.
        raw_name = unquote(blob_name)
        filename = raw_name.rsplit("/", 1)[-1]
        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()

        content_type = _EXT_CONTENT_TYPE.get(ext)

        # Force inline so browsers / Office Online preview instead of
        # downloading. RFC 5987 encoding keeps non-ASCII filenames safe.
        safe = ( 
          filename.replace('"', "")
          .replace("–", "-")
          .replace("—", "-")
        )  
        content_disposition = f'inline; filename="{safe}"'

        return content_type, content_disposition

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _parse(split) -> Tuple[str, str, str]:
        host = split.netloc.split(".", 1)[0]
        path = split.path.lstrip("/")
        if "/" not in path:
            return host, "", ""
        container, blob_name = path.split("/", 1)
        return host, container, blob_name

    def _account_key_from_connection_string(self) -> Optional[str]:
        if not self._connection_string:
            return None
        for part in self._connection_string.split(";"):
            if part.lower().startswith("accountkey="):
                return part.split("=", 1)[1]
        return None

    def _get_user_delegation_key(self, account_name: str) -> UserDelegationKey:
        now = datetime.now(timezone.utc)
        with self._lock:
            cached = self._udk_cache.get(account_name)
            if cached and cached[1] > now + _UDK_REFRESH_SKEW:
                return cached[0]

            account_url = f"https://{account_name}{_BLOB_HOST_SUFFIX}"
            client = BlobServiceClient(
                account_url=account_url, credential=self._credential
            )
            start = now - timedelta(minutes=5)
            end = now + timedelta(hours=_UDK_REQUEST_HOURS)
            udk = client.get_user_delegation_key(
                key_start_time=start, key_expiry_time=end
            )
            self._udk_cache[account_name] = (udk, end)
            return udk


# --------------------------------------------------------------------- module
_signer: Optional[BlobUrlSigner] = None
_signer_lock = threading.Lock()


def _build_signer() -> BlobUrlSigner:
    use_mi = os.getenv("USE_MANAGED_IDENTITY", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    conn = os.getenv("BLOBSTORAGE_CONNECTION_STRING") or os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    return BlobUrlSigner(use_managed_identity=use_mi, connection_string=conn)


def get_blob_signer() -> BlobUrlSigner:
    """Return a process-wide :class:`BlobUrlSigner`."""
    global _signer
    if _signer is None:
        with _signer_lock:
            if _signer is None:
                _signer = _build_signer()
    return _signer


def sign_blob_url(url: Optional[str]) -> Optional[str]:
    """Convenience wrapper. Never raises; returns the input unchanged on error."""
    if not url:
        return url
    try:
        return get_blob_signer().sign(url)
    except Exception:
        return url
