"""Renew an authenticated KA token without changing the SAML login flow."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.utils.saml import create_jwt_token


router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/refresh")
async def refresh_token(request: Request):
    # The API middleware has already verified signature, expiry and identity.
    # Never accept identity/roles from a request body or an expired token.
    principal = request.state.authenticated_user
    access_token = create_jwt_token(principal.user_data)
    return JSONResponse(
        {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
        },
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )

