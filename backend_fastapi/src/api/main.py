import os
import logging
import secrets
from typing import Dict, Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App initialization and configuration
# -----------------------------------------------------------------------------

# PUBLIC_INTERFACE
def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Internal helper to get environment variables with optional default."""
    return os.getenv(name, default)


APP_TITLE = "OceanConnect Backend API"
APP_DESCRIPTION = (
    "FastAPI backend for Jira authentication and data proxy. "
    "Tokens stored in-memory only for demo."
)
APP_VERSION = "0.2.0"

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    contact={"name": "OceanConnect", "url": "https://example.com"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Auth", "description": "OAuth 2.0 (3LO) for Jira"},
        {"name": "Jira", "description": "Jira data endpoints"},
    ],
)

# CORS setup - default allow http://localhost:3000
allowed_origins = (_get_env("ALLOWED_ORIGINS") or "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("oceanconnect.backend")
_log_level = _get_env("LOG_LEVEL", "INFO") or "INFO"
logging.basicConfig(
    level=getattr(logging, _log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)

# -----------------------------------------------------------------------------
# In-memory session storage
# -----------------------------------------------------------------------------
# Tokens stored by state/session_id. No persistence.
SessionTokens = Dict[str, Dict[str, Any]]
SESSIONS: SessionTokens = {}

# -----------------------------------------------------------------------------
# Constants and helpers
# -----------------------------------------------------------------------------
ATLASSIAN_AUTH_BASE = "https://auth.atlassian.com"
ATLASSIAN_API_BASE = "https://api.atlassian.com"

# Expected envs (placeholders ok for demo)
# JIRA_OAUTH_CLIENT_ID, JIRA_OAUTH_CLIENT_SECRET, JIRA_OAUTH_REDIRECT_URI
# ALLOWED_ORIGINS, FRONTEND_REDIRECT_AFTER_AUTH

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error detail message")

class OAuthInitResponse(BaseModel):
    auth_url: str = Field(..., description="URL to redirect user to Atlassian for consent")
    state: str = Field(..., description="State value to validate in callback")

class TokenResponse(BaseModel):
    access_token: str = Field(..., description="Access token issued by Atlassian")
    scope: Optional[str] = Field(None, description="Granted scopes")
    expires_in: Optional[int] = Field(None, description="Seconds until expiry")
    token_type: Optional[str] = Field(None, description="Token type, usually Bearer")
    refresh_token: Optional[str] = Field(None, description="Refresh token if offline_access granted")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_session_by_state(state: str) -> Dict[str, Any]:
    return SESSIONS.setdefault(state, {})

def _store_state(state: str):
    _ensure_session_by_state(state)
    SESSIONS[state]["state"] = state

def _store_token_by_state(state: str, token: Dict[str, Any]):
    session = _ensure_session_by_state(state)
    session["jira_token"] = token

def _get_token_by_state(state: str) -> Dict[str, Any]:
    session = SESSIONS.get(state) or {}
    token = session.get("jira_token")
    if not token:
        raise HTTPException(status_code=401, detail="No Jira token for this session/state")
    return token

async def _http_get(url: str, headers: Dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.get(url, headers=headers)

async def _http_post(url: str, data: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.post(url, data=data, headers=headers)

# -----------------------------------------------------------------------------
# Routes - Health
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get("/", tags=["Auth"], summary="Health Check")
def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Routes - OAuth
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get(
    "/auth/initiate",
    response_model=OAuthInitResponse,
    tags=["Auth"],
    summary="Initiate Jira OAuth 2.0 (3LO) and return presigned authorization URL",
    responses={400: {"model": ErrorResponse}},
)
def auth_initiate() -> OAuthInitResponse:
    """
    Generate a pre-signed authorization URL for Jira OAuth 2.0 (3LO).
    Uses env variables for client config. Stores generated state in memory.
    """
    client_id = _get_env("JIRA_OAUTH_CLIENT_ID") or "YOUR_JIRA_CLIENT_ID"
    redirect_uri = _get_env("JIRA_OAUTH_REDIRECT_URI") or "http://localhost:8000/auth/callback"
    scope = _get_env("JIRA_OAUTH_SCOPE") or "read:jira-user read:jira-work read:me offline_access"

    state = secrets.token_urlsafe(24)
    _store_state(state)

    # Properly encode scope
    qp = httpx.QueryParams({"scope": scope})
    scope_encoded = qp.get("scope")

    auth_url = (
        f"{ATLASSIAN_AUTH_BASE}/authorize"
        f"?audience=api.atlassian.com"
        f"&client_id={client_id}"
        f"&scope={scope_encoded}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&response_type=code"
        f"&prompt=consent"
    )

    logger.info(f"[AUTH] Initiated OAuth; state={state}")
    return OAuthInitResponse(auth_url=auth_url, state=state)

# PUBLIC_INTERFACE
@app.get(
    "/auth/callback",
    tags=["Auth"],
    summary="Handle Jira OAuth callback and exchange code for token",
    responses={200: {"description": "Token JSON or success message"}, 400: {"model": ErrorResponse}},
)
async def auth_callback(
    code: str = Query(..., description="Authorization code from Atlassian"),
    state: str = Query(..., description="Opaque state value"),
) -> Dict[str, Any]:
    """
    Accepts authorization code and state, exchanges code for access token at Atlassian OAuth endpoint,
    stores token in memory keyed by state, and returns token JSON (for demo).
    """
    client_id = _get_env("JIRA_OAUTH_CLIENT_ID") or "YOUR_JIRA_CLIENT_ID"
    client_secret = _get_env("JIRA_OAUTH_CLIENT_SECRET") or "YOUR_JIRA_CLIENT_SECRET"
    redirect_uri = _get_env("JIRA_OAUTH_REDIRECT_URI") or "http://localhost:8000/auth/callback"

    session = SESSIONS.get(state)
    if not session or session.get("state") != state:
        logger.warning(f"[AUTH] Invalid or unknown state received: {state}")
        raise HTTPException(status_code=400, detail="Invalid state")

    token_url = f"{ATLASSIAN_AUTH_BASE}/oauth/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }

    logger.info(f"[AUTH] Exchanging code for token; state={state}")
    resp = await _http_post(token_url, data=data, headers=headers)
    if resp.status_code != 200:
        logger.error(f"[AUTH] Token exchange failed: {resp.text}")
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code for token")

    token = resp.json()
    _store_token_by_state(state, token)
    logger.info(f"[AUTH] Token stored; state={state}")

    # For demo, we return token JSON. In production, redirect to frontend and store only server-side.
    return token

# -----------------------------------------------------------------------------
# Routes - Jira protected resource
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get(
    "/api/jira/projects",
    tags=["Jira"],
    summary="List Jira projects using stored access token",
    responses={200: {"description": "List of projects"}, 401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def jira_projects(
    state: str = Query(..., description="State/session key returned from /auth/initiate"),
) -> Dict[str, Any]:
    """
    Calls Jira Cloud API using stored OAuth token to list projects.
    For demo simplicity, returns the raw payload normalized to { projects: [...] }.
    """
    token = _get_token_by_state(state)
    access_token = token.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing access token")

    # Discover accessible resources to get cloudid
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    resources_url = f"{ATLASSIAN_API_BASE}/oauth/token/accessible-resources"
    logger.info(f"[JIRA] Fetching accessible resources; state={state}")
    res = await _http_get(resources_url, headers=headers)
    if res.status_code != 200:
        logger.error(f"[JIRA] accessible-resources failed: {res.text}")
        raise HTTPException(status_code=400, detail="Failed to fetch accessible resources")

    resources = res.json()
    jira_site = next(iter(resources), None)
    if not jira_site or not jira_site.get("id"):
        raise HTTPException(status_code=400, detail="No accessible Jira site found")

    cloudid = jira_site["id"]
    url = f"{ATLASSIAN_API_BASE}/ex/jira/{cloudid}/rest/api/3/project/search"
    logger.info(f"[JIRA] Fetching projects; cloudid={cloudid} state={state}")
    resp = await _http_get(url, headers=headers)
    if resp.status_code != 200:
        logger.error(f"[JIRA] Projects fetch failed: {resp.text}")
        raise HTTPException(status_code=400, detail="Failed to fetch Jira projects")

    data = resp.json()
    values = data.get("values") or data.get("projects") or []
    projects = [
        {
            "id": str(p.get("id") or p.get("projectId") or ""),
            "key": p.get("key", ""),
            "name": p.get("name", ""),
        }
        for p in values
    ]
    return {"projects": projects}
