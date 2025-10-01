import os
import base64
import logging
import secrets
from typing import Dict, Optional, Any, List, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
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
    "FastAPI backend for Jira and Confluence authentication and data proxy. "
    "Supports OAuth 2.0 (3LO) and API token flows. Tokens stored in-memory only."
)
APP_VERSION = "0.1.0"

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    contact={"name": "OceanConnect", "url": "https://example.com"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Health", "description": "Basic health endpoints"},
        {"name": "Auth - Jira", "description": "Authenticate and manage Jira tokens."},
        {"name": "Auth - Confluence", "description": "Authenticate and manage Confluence tokens."},
        {"name": "Jira", "description": "Jira data endpoints"},
        {"name": "Confluence", "description": "Confluence data endpoints"},
        {"name": "WebSocket", "description": "Real-time connection info"},
    ],
)

# CORS setup - environment configurable
FRONTEND_ORIGINS = _get_env("ALLOWED_ORIGINS", "*")
allow_origins = [o.strip() for o in FRONTEND_ORIGINS.split(",")] if FRONTEND_ORIGINS else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("oceanconnect.backend")
_log_level = _get_env("LOG_LEVEL", "INFO") or "INFO"
logging.basicConfig(level=getattr(logging, _log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# In-memory session storage
# -----------------------------------------------------------------------------
# Tokens stored by session_id. No persistence.
SessionTokens = Dict[str, Dict[str, Any]]
SESSIONS: SessionTokens = {}

# -----------------------------------------------------------------------------
# Constants and helpers
# -----------------------------------------------------------------------------
ATLASSIAN_AUTH_BASE = "https://auth.atlassian.com"
ATLASSIAN_API_BASE = "https://api.atlassian.com"
ATLASSIAN_CLOUD_API = "https://your-domain.atlassian.net"  # used for PAT/basic token domain; must be passed by client

# Environment variables needed (documented for orchestrator to set)
# JIRA_OAUTH_CLIENT_ID, JIRA_OAUTH_CLIENT_SECRET, JIRA_OAUTH_REDIRECT_URI
# CONFLUENCE_OAUTH_CLIENT_ID, CONFLUENCE_OAUTH_CLIENT_SECRET, CONFLUENCE_OAUTH_REDIRECT_URI
# ALLOWED_ORIGINS

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error detail message")

class SessionContext(BaseModel):
    session_id: str = Field(..., description="Client provided session or context identifier")

class OAuthStartRequest(SessionContext):
    scope: Optional[str] = Field(default=None, description="Optional space separated scopes override")
    state: Optional[str] = Field(default=None, description="Optional state value (if not provided server generates)")

class OAuthStartResponse(BaseModel):
    auth_url: str = Field(..., description="URL to redirect user to Atlassian for consent")
    state: str = Field(..., description="State value to validate in callback")

class OAuthCallbackQuery(BaseModel):
    code: str = Field(..., description="Authorization code from Atlassian")
    state: str = Field(..., description="Opaque state value returned from start")

class GenericSuccess(BaseModel):
    success: bool = Field(..., description="True if operation succeeded")
    message: Optional[str] = Field(default=None, description="Informational message")

class ApiTokenAuthRequest(SessionContext):
    base_url: str = Field(..., description="Base URL of the Atlassian cloud site e.g. https://your-domain.atlassian.net")
    email: str = Field(..., description="User email for basic auth (PAT)")
    api_token: str = Field(..., description="API token / Personal Access Token")

class JiraProject(BaseModel):
    key: str = Field(..., description="Project key")
    name: str = Field(..., description="Project name")
    id: str = Field(..., description="Project id")

class JiraProjectsResponse(BaseModel):
    projects: List[JiraProject] = Field(..., description="List of projects")

class ConfluenceSpace(BaseModel):
    key: str = Field(..., description="Space key")
    name: str = Field(..., description="Space name")
    id: str = Field(..., description="Space id")

class ConfluenceSpacesResponse(BaseModel):
    spaces: List[ConfluenceSpace] = Field(..., description="List of spaces")

class ConnectionStatus(BaseModel):
    jira_connected: bool = Field(..., description="Whether Jira is connected in this session")
    confluence_connected: bool = Field(..., description="Whether Confluence is connected in this session")
    sites: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional list of accessible Atlassian sites")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "jira": {},
            "confluence": {},
        }
    return SESSIONS[session_id]

def _store_state(session_id: str, provider: str, state: str):
    session = _ensure_session(session_id)
    session[provider]["oauth_state"] = state

def _validate_state(session_id: str, provider: str, returned_state: str):
    session = _ensure_session(session_id)
    expected = session.get(provider, {}).get("oauth_state")
    if not expected or expected != returned_state:
        raise HTTPException(status_code=400, detail="Invalid state for OAuth callback")

def _store_token(session_id: str, provider: str, token: Dict[str, Any]):
    session = _ensure_session(session_id)
    session[provider]["token"] = token

def _store_api_token(session_id: str, provider: str, base_url: str, email: str, api_token: str):
    session = _ensure_session(session_id)
    session[provider]["basic"] = {
        "base_url": base_url.rstrip("/"),
        "email": email,
        "api_token": api_token,
    }

def _get_jira_headers(session_data: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    """Return headers and base url for Jira based on stored creds."""
    jira_data = session_data.get("jira", {})
    if "token" in jira_data:
        access_token = jira_data["token"].get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Jira OAuth token missing")
        return {"Authorization": f"Bearer {access_token}"}, ATLASSIAN_API_BASE
    elif "basic" in jira_data:
        bd = jira_data["basic"]
        cred = base64.b64encode(f"{bd['email']}:{bd['api_token']}".encode()).decode()
        return {"Authorization": f"Basic {cred}"}, bd["base_url"]
    else:
        raise HTTPException(status_code=401, detail="Jira not connected")

def _get_confluence_headers(session_data: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    """Return headers and base url for Confluence based on stored creds."""
    conf_data = session_data.get("confluence", {})
    if "token" in conf_data:
        access_token = conf_data["token"].get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="Confluence OAuth token missing")
        return {"Authorization": f"Bearer {access_token}"}, ATLASSIAN_API_BASE
    elif "basic" in conf_data:
        bd = conf_data["basic"]
        cred = base64.b64encode(f"{bd['email']}:{bd['api_token']}".encode()).decode()
        return {"Authorization": f"Basic {cred}"}, bd["base_url"]
    else:
        raise HTTPException(status_code=401, detail="Confluence not connected")

async def _http_get(url: str, headers: Dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        return resp

async def _http_post(url: str, data: Dict[str, Any], headers: Dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, data=data, headers=headers)
        return resp

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Health Check")
def health_check() -> Dict[str, str]:
    """Return simple health check message."""
    return {"message": "Healthy"}

# -----------------------------------------------------------------------------
# Connection status
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get("/status", response_model=ConnectionStatus, tags=["Health"], summary="Connection status by session")
def connection_status(session_id: str = Query(..., description="Session identifier to query")):
    """
    Get connection status for Jira and Confluence for a given session.
    """
    session = _ensure_session(session_id)
    return ConnectionStatus(
        jira_connected=bool(session.get("jira", {}).get("token") or session.get("jira", {}).get("basic")),
        confluence_connected=bool(session.get("confluence", {}).get("token") or session.get("confluence", {}).get("basic")),
        sites=None,
    )

# -----------------------------------------------------------------------------
# JIRA AUTH - OAuth
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.post(
    "/auth/jira/oauth/start",
    response_model=OAuthStartResponse,
    tags=["Auth - Jira"],
    summary="Start Jira OAuth 2.0 (3LO) flow",
    responses={400: {"model": ErrorResponse}},
)
def jira_oauth_start(req: OAuthStartRequest):
    """
    Begin Jira OAuth 2.0 (3LO) flow by generating a state and redirect URL.

    Request body:
    - session_id: Client session identifier
    - scope: Optional space-delimited scope override
    - state: Optional state override (server will generate if not provided)

    Returns:
    - auth_url to redirect user to Atlassian
    - state used for CSRF protection
    """
    client_id = _get_env("JIRA_OAUTH_CLIENT_ID")
    redirect_uri = _get_env("JIRA_OAUTH_REDIRECT_URI")
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=400, detail="JIRA OAuth not configured")

    scope = req.scope or "read:jira-user read:jira-work read:me offline_access"
    state = req.state or secrets.token_urlsafe(24)
    _store_state(req.session_id, "jira", state)

    auth_url = (
        f"{ATLASSIAN_AUTH_BASE}/authorize"
        f"?audience=api.atlassian.com"
        f"&client_id={client_id}"
        f"&scope={httpx.QueryParams({'scope': scope}).get('scope')}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&response_type=code"
        f"&prompt=consent"
    )
    logger.info(f"[JIRA][{req.session_id}] OAuth start initiated")
    return OAuthStartResponse(auth_url=auth_url, state=state)

# PUBLIC_INTERFACE
@app.get(
    "/auth/jira/oauth/callback",
    tags=["Auth - Jira"],
    summary="Handle Jira OAuth callback",
    responses={302: {"description": "Redirect to frontend"}, 400: {"model": ErrorResponse}},
)
async def jira_oauth_callback(
    request: Request,
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State value"),
    session_id: str = Query(..., description="Session identifier to map tokens"),
):
    """
    Handle Jira OAuth callback: exchanges code for tokens and stores them in-memory keyed by session_id.
    Requires query params: code, state, session_id.

    Redirects back to the frontend redirect URI (if provided via env FRONTEND_REDIRECT_AFTER_AUTH) with status.
    """
    client_id = _get_env("JIRA_OAUTH_CLIENT_ID")
    client_secret = _get_env("JIRA_OAUTH_CLIENT_SECRET")
    redirect_uri = _get_env("JIRA_OAUTH_REDIRECT_URI")
    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=400, detail="JIRA OAuth not configured")

    _validate_state(session_id, "jira", state)

    token_url = f"{ATLASSIAN_AUTH_BASE}/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    logger.info(f"[JIRA][{session_id}] Exchanging code for token")
    resp = await _http_post(token_url, data=data, headers=headers)
    if resp.status_code != 200:
        logger.error(f"[JIRA][{session_id}] Token exchange failed: {resp.text}")
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code for token")

    token_payload = resp.json()
    _store_token(session_id, "jira", token_payload)
    logger.info(f"[JIRA][{session_id}] OAuth token stored")

    redirect_after = _get_env("FRONTEND_REDIRECT_AFTER_AUTH")
    if redirect_after:
        url = f"{redirect_after}?provider=jira&status=success"
        return RedirectResponse(url=url, status_code=302)
    return {"success": True, "provider": "jira"}

# -----------------------------------------------------------------------------
# JIRA AUTH - API Token (Basic)
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.post(
    "/auth/jira/api-token",
    response_model=GenericSuccess,
    tags=["Auth - Jira"],
    summary="Connect Jira using API token (basic auth)",
    responses={400: {"model": ErrorResponse}},
)
async def jira_api_token(req: ApiTokenAuthRequest):
    """
    Store Jira API token credentials and validate by calling a simple Jira endpoint.
    """
    headers = {
        "Authorization": "Basic "
        + base64.b64encode(f"{req.email}:{req.api_token}".encode()).decode(),
        "Accept": "application/json",
    }
    # Validate by fetching projects (minimal route)
    url = f"{req.base_url.rstrip('/')}/rest/api/3/project/search"
    logger.info(f"[JIRA][{req.session_id}] Validating API token at {url}")
    resp = await _http_get(url, headers=headers)
    if resp.status_code not in (200, 401, 403):
        # Unexpected code
        logger.error(f"[JIRA][{req.session_id}] Validation failed status={resp.status_code} body={resp.text}")
        raise HTTPException(status_code=400, detail="Unable to validate Jira token (unexpected response)")

    if resp.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid Jira API token or permissions")

    _store_api_token(req.session_id, "jira", req.base_url, req.email, req.api_token)
    logger.info(f"[JIRA][{req.session_id}] API token stored")
    return GenericSuccess(success=True, message="Jira connected")

# -----------------------------------------------------------------------------
# CONFLUENCE AUTH - OAuth
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.post(
    "/auth/confluence/oauth/start",
    response_model=OAuthStartResponse,
    tags=["Auth - Confluence"],
    summary="Start Confluence OAuth 2.0 (3LO) flow",
    responses={400: {"model": ErrorResponse}},
)
def confluence_oauth_start(req: OAuthStartRequest):
    """
    Begin Confluence OAuth 2.0 (3LO) flow by generating a state and redirect URL.
    """
    client_id = _get_env("CONFLUENCE_OAUTH_CLIENT_ID")
    redirect_uri = _get_env("CONFLUENCE_OAUTH_REDIRECT_URI")
    if not client_id or not redirect_uri:
        raise HTTPException(status_code=400, detail="Confluence OAuth not configured")

    scope = req.scope or "read:confluence-space.summary read:confluence-content.summary read:me offline_access"
    state = req.state or secrets.token_urlsafe(24)
    _store_state(req.session_id, "confluence", state)

    auth_url = (
        f"{ATLASSIAN_AUTH_BASE}/authorize"
        f"?audience=api.atlassian.com"
        f"&client_id={client_id}"
        f"&scope={httpx.QueryParams({'scope': scope}).get('scope')}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        f"&response_type=code"
        f"&prompt=consent"
    )
    logger.info(f"[CONFLUENCE][{req.session_id}] OAuth start initiated")
    return OAuthStartResponse(auth_url=auth_url, state=state)

# PUBLIC_INTERFACE
@app.get(
    "/auth/confluence/oauth/callback",
    tags=["Auth - Confluence"],
    summary="Handle Confluence OAuth callback",
    responses={302: {"description": "Redirect to frontend"}, 400: {"model": ErrorResponse}},
)
async def confluence_oauth_callback(
    request: Request,
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State value"),
    session_id: str = Query(..., description="Session identifier to map tokens"),
):
    """
    Handle Confluence OAuth callback: exchanges code for tokens and stores them in-memory keyed by session_id.
    Requires query params: code, state, session_id.
    """
    client_id = _get_env("CONFLUENCE_OAUTH_CLIENT_ID")
    client_secret = _get_env("CONFLUENCE_OAUTH_CLIENT_SECRET")
    redirect_uri = _get_env("CONFLUENCE_OAUTH_REDIRECT_URI")
    if not client_id or not client_secret or not redirect_uri:
        raise HTTPException(status_code=400, detail="Confluence OAuth not configured")

    _validate_state(session_id, "confluence", state)

    token_url = f"{ATLASSIAN_AUTH_BASE}/oauth/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    logger.info(f"[CONFLUENCE][{session_id}] Exchanging code for token")
    resp = await _http_post(token_url, data=data, headers=headers)
    if resp.status_code != 200:
        logger.error(f"[CONFLUENCE][{session_id}] Token exchange failed: {resp.text}")
        raise HTTPException(status_code=400, detail="Failed to exchange authorization code for token")

    token_payload = resp.json()
    _store_token(session_id, "confluence", token_payload)
    logger.info(f"[CONFLUENCE][{session_id}] OAuth token stored")

    redirect_after = _get_env("FRONTEND_REDIRECT_AFTER_AUTH")
    if redirect_after:
        url = f"{redirect_after}?provider=confluence&status=success"
        return RedirectResponse(url=url, status_code=302)
    return {"success": True, "provider": "confluence"}

# -----------------------------------------------------------------------------
# CONFLUENCE AUTH - API Token (Basic)
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.post(
    "/auth/confluence/api-token",
    response_model=GenericSuccess,
    tags=["Auth - Confluence"],
    summary="Connect Confluence using API token (basic auth)",
    responses={400: {"model": ErrorResponse}},
)
async def confluence_api_token(req: ApiTokenAuthRequest):
    """
    Store Confluence API token credentials and validate by calling a simple Confluence endpoint.
    """
    headers = {
        "Authorization": "Basic "
        + base64.b64encode(f"{req.email}:{req.api_token}".encode()).decode(),
        "Accept": "application/json",
    }
    url = f"{req.base_url.rstrip('/')}/wiki/api/v2/spaces"
    logger.info(f"[CONFLUENCE][{req.session_id}] Validating API token at {url}")
    resp = await _http_get(url, headers=headers)
    if resp.status_code not in (200, 401, 403):
        logger.error(f"[CONFLUENCE][{req.session_id}] Validation failed status={resp.status_code} body={resp.text}")
        raise HTTPException(status_code=400, detail="Unable to validate Confluence token (unexpected response)")

    if resp.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Invalid Confluence API token or permissions")

    _store_api_token(req.session_id, "confluence", req.base_url, req.email, req.api_token)
    logger.info(f"[CONFLUENCE][{req.session_id}] API token stored")
    return GenericSuccess(success=True, message="Confluence connected")

# -----------------------------------------------------------------------------
# Jira data endpoints
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get(
    "/jira/projects",
    response_model=JiraProjectsResponse,
    tags=["Jira"],
    summary="List Jira projects",
    responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def jira_projects(session_id: str = Query(..., description="Session identifier to use stored credentials")):
    """
    Returns list of Jira projects for the connected account in this session.

    If OAuth is used, it calls the Atlassian cloud API via gateway.
    If API token is used, it calls the site's REST endpoint directly.
    """
    session = _ensure_session(session_id)
    headers, base_url = _get_jira_headers(session)

    # When using OAuth (Bearer to api.atlassian.com), we must discover accessible resources (cloudid)
    if base_url == ATLASSIAN_API_BASE:
        # Get accessible resources
        me_headers = headers.copy()
        me_headers["Accept"] = "application/json"
        logger.info(f"[JIRA][{session_id}] Fetching accessible resources")
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(f"{ATLASSIAN_API_BASE}/oauth/token/accessible-resources", headers=me_headers)
        if res.status_code != 200:
            logger.error(f"[JIRA][{session_id}] Failed accessible-resources: {res.text}")
            raise HTTPException(status_code=400, detail="Failed to fetch accessible resources")

        resources = res.json()
        # choose first jira resource
        jira_site = next((r for r in resources if "jira" in (r.get("scopes") or []) or r.get("name")), None)
        cloudid = jira_site.get("id") if jira_site else None
        if not cloudid:
            raise HTTPException(status_code=400, detail="No accessible Jira site found for this account")

        url = f"{ATLASSIAN_API_BASE}/ex/jira/{cloudid}/rest/api/3/project/search"
        logger.info(f"[JIRA][{session_id}] Fetching projects via cloud id {cloudid}")
        resp = await _http_get(url, headers=headers)
        if resp.status_code != 200:
            logger.error(f"[JIRA][{session_id}] Projects fetch failed: {resp.text}")
            raise HTTPException(status_code=400, detail="Failed to fetch Jira projects")
        data = resp.json()
        values = data.get("values") or data.get("projects") or []
    else:
        # Basic auth direct site call
        url = f"{base_url}/rest/api/3/project/search"
        logger.info(f"[JIRA][{session_id}] Fetching projects from {url}")
        resp = await _http_get(url, headers=headers)
        if resp.status_code != 200:
            logger.error(f"[JIRA][{session_id}] Projects fetch failed: {resp.text}")
            raise HTTPException(status_code=400, detail="Failed to fetch Jira projects")
        data = resp.json()
        values = data.get("values") or data.get("projects") or []

    projects: List[JiraProject] = []
    for p in values:
        # Jira returns different shapes; normalize
        pid = str(p.get("id") or p.get("projectId") or "")
        projects.append(JiraProject(id=pid, key=p.get("key", ""), name=p.get("name", "")))

    return JiraProjectsResponse(projects=projects)

# -----------------------------------------------------------------------------
# Confluence data endpoints
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get(
    "/confluence/spaces",
    response_model=ConfluenceSpacesResponse,
    tags=["Confluence"],
    summary="List Confluence spaces",
    responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def confluence_spaces(session_id: str = Query(..., description="Session identifier to use stored credentials")):
    """
    Returns list of Confluence spaces for the connected account in this session.
    """
    session = _ensure_session(session_id)
    headers, base_url = _get_confluence_headers(session)

    if base_url == ATLASSIAN_API_BASE:
        # Discover resources to get cloudid and baseUrl for wiki
        me_headers = headers.copy()
        me_headers["Accept"] = "application/json"
        logger.info(f"[CONFLUENCE][{session_id}] Fetching accessible resources")
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.get(f"{ATLASSIAN_API_BASE}/oauth/token/accessible-resources", headers=me_headers)
        if res.status_code != 200:
            logger.error(f"[CONFLUENCE][{session_id}] Failed accessible-resources: {res.text}")
            raise HTTPException(status_code=400, detail="Failed to fetch accessible resources")

        resources = res.json()
        conf_site = next((r for r in resources if r.get("url", "").endswith(".atlassian.net")), None)
        base_wiki = (conf_site or {}).get("url") or ""
        if not base_wiki:
            raise HTTPException(status_code=400, detail="No accessible Confluence site found for this account")

        url = f"{base_wiki.rstrip('/')}/wiki/api/v2/spaces"
        logger.info(f"[CONFLUENCE][{session_id}] Fetching spaces from {url}")
        resp = await _http_get(url, headers=headers)
        if resp.status_code != 200:
            logger.error(f"[CONFLUENCE][{session_id}] Spaces fetch failed: {resp.text}")
            raise HTTPException(status_code=400, detail="Failed to fetch Confluence spaces")
        data = resp.json()
        results = data.get("results") or data.get("data") or []
    else:
        url = f"{base_url}/wiki/api/v2/spaces"
        logger.info(f"[CONFLUENCE][{session_id}] Fetching spaces from {url}")
        resp = await _http_get(url, headers=headers)
        if resp.status_code != 200:
            logger.error(f"[CONFLUENCE][{session_id}] Spaces fetch failed: {resp.text}")
            raise HTTPException(status_code=400, detail="Failed to fetch Confluence spaces")
        data = resp.json()
        results = data.get("results") or data.get("data") or []

    spaces: List[ConfluenceSpace] = []
    for s in results:
        sid = str(s.get("id", ""))
        key = s.get("key", "") or s.get("spaceKey", "")
        name = s.get("name", "") or s.get("displayName", "")
        spaces.append(ConfluenceSpace(id=sid, key=key, name=name))

    return ConfluenceSpacesResponse(spaces=spaces)

# -----------------------------------------------------------------------------
# WebSocket docs helper (no actual ws endpoints implemented here, but doc stub)
# -----------------------------------------------------------------------------
# PUBLIC_INTERFACE
@app.get(
    "/websocket-docs",
    tags=["WebSocket"],
    summary="WebSocket usage",
    description="This project currently does not expose real-time WebSockets, but this endpoint documents future usage.",
)
def websocket_docs() -> Dict[str, str]:
    """
    WebSocket usage note: Real-time features may be added later. No active WS endpoints now.
    """
    return {
        "message": "No WebSocket endpoints at the moment. Use REST endpoints for Jira/Confluence data.",
        "note": "If real-time updates are needed, add ws routes and tag accordingly.",
    }
