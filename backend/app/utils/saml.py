import os
import json
import datetime
import jwt  # PyJWT
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, JSONResponse
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

# Configuration
admin_group_id = os.getenv('ADMIN_GROUP_ID')
redirect_url = os.getenv('REDIRECT_URL')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')

# SAML path points to backend/app/saml/ directory
SAML_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saml")

router = APIRouter(prefix="/saml", tags=["SAML"])


# -------------------------
#   INIT SAML AUTH
# -------------------------
def init_saml_auth(req, saml_path):
    print('In init auth')
    return OneLogin_Saml2_Auth(req, custom_base_path=saml_path)


# -------------------------
#   PREPARE FASTAPI REQUEST
# -------------------------
async def prepare_fastapi_request(request: Request):
    print('In Prepare FastAPI Request')

    try:
        form = await request.form()
        post_data = dict(form)
    except Exception:
        post_data = {}

    get_data = dict(request.query_params)

    # Azure sends the real domain in x-forwarded-host
    host = request.headers.get("x-forwarded-host", request.headers.get("host"))

    return {
        'https': 'on',
        'http_host': host,
        'script_name': request.url.path,
        'server_port': '443',
        'get_data': get_data,
        'post_data': post_data
    }


# -------------------------
#   JWT CREATION
# -------------------------
def create_jwt_token(user_data):
    expiration = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    payload = {
        'user_data': user_data,
        'exp': expiration
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')
    return token


# -------------------------
#   JWT DECODING
# -------------------------
def get_data_from_token(token):
    try:
        decoded_data = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return decoded_data.get('user_data')
    except ExpiredSignatureError:
        return 'Error: Token has expired'
    except InvalidTokenError:
        return 'Error: Invalid token'


# -------------------------
#   SAML LOGIN
# -------------------------
async def saml_login(request: Request, saml_path):
    try:
        print('In SAML Login')
        req = await prepare_fastapi_request(request)
        print(f'Request Prepared: {req}')
        auth = init_saml_auth(req, saml_path)
        print('SAML Auth Initialized')
        login_url = auth.login()
        print(f'Redirecting to: {login_url}')
        return RedirectResponse(login_url)
    except Exception as e:
        print(f'Error during SAML login: {str(e)}')
        return JSONResponse({"error": str(e)}, status_code=500)


# -------------------------
#   SAML CALLBACK
# -------------------------
async def saml_callback(request: Request, saml_path):
    print('In SAML Callback')
    req = await prepare_fastapi_request(request)
    auth = init_saml_auth(req, saml_path)

    await asyncio.to_thread(auth.process_response)
    errors = auth.get_errors()
    group_name = 'user'

    if not errors:
        user_data_from_saml = auth.get_attributes()
        name_id_from_saml = auth.get_nameid()

        # FastAPI has no session. Just store local variables.
        json_data = user_data_from_saml

        groups = json_data.get("http://schemas.microsoft.com/ws/2008/06/identity/claims/groups", [])

        if admin_group_id and admin_group_id in groups:
            group_name = 'admin'

        user_data = {
            'name': json_data.get('http://schemas.microsoft.com/identity/claims/displayname'),
            'group': group_name,
            'job_title': json_data.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/userjobtitle', [None])[0],
            'email': json_data.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress')
        }

        # Save as before
        def write_session_file():
            with open("session_data_from_backend.txt", "w") as f:
                f.write(json.dumps(json_data, indent=4))

        await asyncio.to_thread(write_session_file)

        token = create_jwt_token(user_data)
        return RedirectResponse(f"{redirect_url}?token={token}", status_code=303)

    else:
        return JSONResponse(
            {"error": f"SAML Authentication failed: {errors}", "request": req},
            status_code=500
        )


# -------------------------
#   TOKEN EXTRACTOR
# -------------------------
async def extract_token(request: Request):
    token = request.query_params.get('token')
    if not token:
        return JSONResponse({"error": "Token is missing"}, status_code=400)

    user_data = get_data_from_token(token)

    if isinstance(user_data, str) and user_data.startswith("Error"):
        return JSONResponse({"error": user_data}, status_code=400)

    return JSONResponse({"user_data": user_data}, status_code=200)


# ---------------------------
# SAML Routes
# ---------------------------

@router.get("/login")
async def login(request: Request):
    return await saml_login(request, SAML_PATH)

@router.post("/callback")
async def login_callback(request: Request):
    return await saml_callback(request, SAML_PATH)

@router.get("/token/extract")
async def func_get_data_from_token(request: Request):
    return await extract_token(request)
    return await extract_token()
