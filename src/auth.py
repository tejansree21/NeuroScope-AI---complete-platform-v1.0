"""
NeuroScope AI — auth.py
Microsoft Entra ID SSO + Session Management + Audit Log + Rate Limiting
"""

import os, json, time, uuid, hashlib, logging, secrets
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import httpx
from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from jose import JWTError, jwt
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger("neuroscope.auth")

# ── Config ─────────────────────────────────────────────────────────────────────
BASE   = os.environ.get("NEUROSCOPE_BASE", "/app/data")
CFGDIR = os.path.join(BASE, "configs")
os.makedirs(CFGDIR, exist_ok=True)

# Microsoft Azure credentials
MS_CLIENT_ID     = os.environ.get("MS_CLIENT_ID",     "fa3f7d31-4341-46d6-a584-482b6f20e008")
MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET", "")   # set via HF env var
MS_TENANT_ID     = os.environ.get("MS_TENANT_ID",     "3030e335-4f27-49ae-b9d1-f36767991c55")
MS_REDIRECT_URI  = os.environ.get("MS_REDIRECT_URI",
                    "https://tejansree-neuroscope-ai.hf.space/auth/microsoft/callback")
MS_AUTHORITY     = f"https://login.microsoftonline.com/{MS_TENANT_ID}"
MS_SCOPES        = ["openid", "profile", "email", "User.Read"]

# JWT
JWT_SECRET    = os.environ.get("NEUROSCOPE_SECRET_KEY", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"

# Session timeouts (seconds of inactivity)
INACTIVITY_TIMEOUT = {
    "superadmin"   : 30 * 60,
    "admin"        : 25 * 60,
    "clinician"    : 20 * 60,
    "viewer"       : 15 * 60,
}
JWT_LIFETIME = {
    "superadmin"   : 8  * 3600,
    "admin"        : 8  * 3600,
    "clinician"    : 8  * 3600,
    "viewer"       : 4  * 3600,
}

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# ── File paths ─────────────────────────────────────────────────────────────────
USERS_FILE     = os.path.join(CFGDIR, "users.json")
SESSIONS_FILE  = os.path.join(CFGDIR, "sessions.json")
HOSPITALS_FILE = os.path.join(CFGDIR, "hospitals.json")
AUDIT_FILE     = os.path.join(CFGDIR, "audit_log.jsonl")
PENDING_FILE   = os.path.join(CFGDIR, "pending_users.json")

# ── Default data structures ────────────────────────────────────────────────────
def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Load error {path}: {e}")
    return default

def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Save error {path}: {e}")

def _load_users():    return _load_json(USERS_FILE,     {})
def _save_users(d):   _save_json(USERS_FILE, d)
def _load_sessions(): return _load_json(SESSIONS_FILE,  {})
def _save_sessions(d):_save_json(SESSIONS_FILE, d)
def _load_hospitals():return _load_json(HOSPITALS_FILE, {})
def _load_pending():  return _load_json(PENDING_FILE,   {})
def _save_pending(d): _save_json(PENDING_FILE, d)

# ── Initial data ───────────────────────────────────────────────────────────────
def _init_defaults():
    """Create default hospitals and superadmin on first run."""
    hospitals = _load_hospitals()
    if not hospitals:
        hospitals = {
            "hospital_a": {
                "name"       : "City General Hospital",
                "tenant_ids" : [],
                "admin_email": "",
                "active"     : True,
                "created_at" : datetime.now().isoformat(),
                "scan_count" : 0,
            },
            "hospital_b": {
                "name"       : "University Medical Centre",
                "tenant_ids" : [],
                "admin_email": "",
                "active"     : True,
                "created_at" : datetime.now().isoformat(),
                "scan_count" : 0,
            },
            "hospital_c": {
                "name"       : "Regional Cancer Centre",
                "tenant_ids" : [],
                "admin_email": "",
                "active"     : True,
                "created_at" : datetime.now().isoformat(),
                "scan_count" : 0,
            },
        }
        _save_json(HOSPITALS_FILE, hospitals)
        logger.info("Default hospitals created")

    users = _load_users()
    if not users:
        users = {
            "superadmin": {
                "email"         : "official.neuroscopeai@gmail.com",
                "microsoft_oid" : None,
                "role"          : "superadmin",
                "hospital_id"   : None,
                "active"        : True,
                "approved"      : True,
                "created_at"    : datetime.now().isoformat(),
                "last_login"    : None,
                "scan_count"    : 0,
                "display_name"  : "NeuroScope Super Admin",
            }
        }
        _save_users(users)
        logger.info("Superadmin user created")

# ── Audit log ──────────────────────────────────────────────────────────────────
def audit(action: str, user: str = "system", detail: str = "",
          ip: str = "", hospital: str = ""):
    entry = {
        "ts"       : datetime.now().isoformat(),
        "action"   : action,
        "user"     : user,
        "detail"   : detail,
        "ip"       : ip,
        "hospital" : hospital,
    }
    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Audit write error: {e}")

# ── JWT helpers ────────────────────────────────────────────────────────────────
def create_jwt(username: str, role: str, hospital_id: Optional[str],
               display_name: str, email: str) -> str:
    lifetime = JWT_LIFETIME.get(role, 8 * 3600)
    payload  = {
        "sub"         : username,
        "role"        : role,
        "hospital_id" : hospital_id,
        "display_name": display_name,
        "email"       : email,
        "exp"         : datetime.utcnow() + timedelta(seconds=lifetime),
        "iat"         : datetime.utcnow(),
        "jti"         : str(uuid.uuid4()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as e:
        raise HTTPException(401, f"Invalid token: {e}")

def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token   = auth[7:]
    payload = decode_jwt(token)

    # Check session still active and inactivity timeout
    sessions = _load_sessions()
    jti      = payload.get("jti", "")
    if jti not in sessions:
        raise HTTPException(401, "Session expired or revoked")

    sess    = sessions[jti]
    timeout = INACTIVITY_TIMEOUT.get(payload["role"], 20 * 60)
    last_at = sess.get("last_activity", sess.get("created_at",""))
    try:
        last_dt = datetime.fromisoformat(last_at)
        if (datetime.now() - last_dt).total_seconds() > timeout:
            # Inactivity timeout — revoke session
            del sessions[jti]
            _save_sessions(sessions)
            audit("session_timeout", payload["sub"])
            raise HTTPException(401, "Session timed out due to inactivity")
    except ValueError:
        pass

    # Update last activity
    sess["last_activity"] = datetime.now().isoformat()
    sessions[jti]         = sess
    _save_sessions(sessions)

    return payload

def require_role(*roles):
    def checker(payload: dict = Depends(get_current_user)):
        if payload["role"] not in roles:
            raise HTTPException(403, f"Requires role: {', '.join(roles)}")
        return payload
    return checker

require_any      = Depends(get_current_user)
require_clinician= Depends(require_role("clinician","admin","superadmin"))
require_admin    = Depends(require_role("admin","superadmin"))
require_super    = Depends(require_role("superadmin"))

# ── Microsoft OAuth helpers ────────────────────────────────────────────────────
def ms_auth_url(state: str) -> str:
    params = {
        "client_id"    : MS_CLIENT_ID,
        "response_type": "code",
        "redirect_uri" : MS_REDIRECT_URI,
        "response_mode": "query",
        "scope"        : " ".join(MS_SCOPES),
        "state"        : state,
        "prompt"       : "select_account",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{MS_AUTHORITY}/oauth2/v2.0/authorize?{query}"

async def ms_exchange_code(code: str) -> dict:
    """Exchange auth code for tokens."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{MS_AUTHORITY}/oauth2/v2.0/token",
            data={
                "client_id"    : MS_CLIENT_ID,
                "client_secret": MS_CLIENT_SECRET,
                "code"         : code,
                "redirect_uri" : MS_REDIRECT_URI,
                "grant_type"   : "authorization_code",
                "scope"        : " ".join(MS_SCOPES),
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise HTTPException(400, f"Token exchange failed: {resp.text}")
        return resp.json()

async def ms_get_user(access_token: str) -> dict:
    """Get user profile from Microsoft Graph."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise HTTPException(400, "Failed to fetch user profile")
        return resp.json()

def detect_hospital_from_email(email: str) -> Optional[str]:
    """
    Match a user's email domain to a registered hospital tenant.
    Falls back to checking hospital tenant_ids list.
    """
    hospitals = _load_hospitals()
    domain    = email.split("@")[-1].lower() if "@" in email else ""
    for h_id, h_data in hospitals.items():
        domains = h_data.get("email_domains", [])
        if domain in domains:
            return h_id
    return None

# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/auth", tags=["auth"])

# ── In-memory PKCE/state store (replace with Redis in production) ──────────────
_state_store: dict = {}

@router.get("/microsoft/login")
@limiter.limit("20/minute")
async def microsoft_login(request: Request):
    """
    Redirect user to Microsoft login page.
    Generates a random state for CSRF protection.
    """
    state              = secrets.token_urlsafe(32)
    _state_store[state]= {"created_at": time.time()}
    login_url          = ms_auth_url(state)
    audit("login_initiated", ip=request.client.host if request.client else "")
    return RedirectResponse(login_url)

@router.get("/microsoft/callback")
async def microsoft_callback(request: Request, code: str = "", state: str = "",
                              error: str = "", error_description: str = ""):
    """
    Microsoft redirects here after authentication.
    Exchange code for token, get user profile, create/update user record.
    """
    if error:
        audit("login_error", detail=error)
        return RedirectResponse(f"/ui?error={error}")

    # Validate state
    if state not in _state_store:
        audit("login_csrf_fail", detail="Invalid state")
        raise HTTPException(400, "Invalid state — possible CSRF attack")
    del _state_store[state]

    # Exchange code for tokens
    try:
        tokens = await ms_exchange_code(code)
    except HTTPException as e:
        audit("login_token_fail", detail=str(e.detail))
        return RedirectResponse(f"/ui?error=token_exchange_failed")

    access_token = tokens.get("access_token", "")
    id_token     = tokens.get("id_token", "")

    # Get Microsoft user profile
    try:
        ms_user = await ms_get_user(access_token)
    except HTTPException:
        return RedirectResponse("/ui?error=profile_fetch_failed")

    email        = ms_user.get("mail") or ms_user.get("userPrincipalName", "")
    display_name = ms_user.get("displayName", email.split("@")[0])
    ms_oid       = ms_user.get("id", "")   # Microsoft Object ID — unique per user

    # Look up or create user record
    users    = _load_users()
    username = None

    # Find by Microsoft OID first (most reliable)
    for uname, udata in users.items():
        if udata.get("microsoft_oid") == ms_oid:
            username = uname
            break

    # Fallback: find by email
    if not username:
        for uname, udata in users.items():
            if udata.get("email","").lower() == email.lower():
                username = uname
                users[username]["microsoft_oid"] = ms_oid
                break

    # Brand new user — create pending record
    if not username:
        username = email.split("@")[0].lower().replace(".","_") + "_" + ms_oid[:6]
        hospital = detect_hospital_from_email(email)
        users[username] = {
            "email"         : email,
            "microsoft_oid" : ms_oid,
            "role"          : "pending",
            "hospital_id"   : hospital,
            "active"        : True,
            "approved"      : False,
            "created_at"    : datetime.now().isoformat(),
            "last_login"    : None,
            "scan_count"    : 0,
            "display_name"  : display_name,
        }
        _save_users(users)
        audit("new_user_registered", user=username, detail=email,
              ip=request.client.host if request.client else "")
        # Redirect to pending approval screen
        return RedirectResponse(f"/ui?status=pending_approval&user={username}")

    user = users[username]

    # Check if user is active
    if not user.get("active", True):
        audit("login_blocked", user=username, detail="account_deactivated")
        return RedirectResponse("/ui?error=account_deactivated")

    # Check if user is approved
    if not user.get("approved", False):
        return RedirectResponse(f"/ui?status=pending_approval&user={username}")

    # Update last login + OID
    users[username]["last_login"]    = datetime.now().isoformat()
    users[username]["microsoft_oid"] = ms_oid
    users[username]["display_name"]  = display_name
    _save_users(users)

    # Create JWT
    role        = user.get("role", "viewer")
    hospital_id = user.get("hospital_id")
    token       = create_jwt(username, role, hospital_id, display_name, email)

    # Decode to get jti
    payload = decode_jwt(token)
    jti     = payload["jti"]

    # Register session
    sessions     = _load_sessions()
    sessions[jti]= {
        "username"      : username,
        "role"          : role,
        "hospital_id"   : hospital_id,
        "ip"            : request.client.host if request.client else "",
        "created_at"    : datetime.now().isoformat(),
        "last_activity" : datetime.now().isoformat(),
    }
    _save_sessions(sessions)

    audit("login_success", user=username, hospital=hospital_id or "",
          ip=request.client.host if request.client else "")

    # Set httpOnly cookie with token + redirect to UI
    response = RedirectResponse(f"/ui?login=success")
    response.set_cookie(
        key="ns_token", value=token,
        httponly=True, secure=True, samesite="lax",
        max_age=JWT_LIFETIME.get(role, 28800),
    )
    return response

@router.post("/microsoft/logout")
async def microsoft_logout(request: Request,
                            payload: dict = Depends(get_current_user)):
    """Invalidate session and clear cookie."""
    jti      = payload.get("jti","")
    sessions = _load_sessions()
    if jti in sessions:
        del sessions[jti]
        _save_sessions(sessions)

    audit("logout", user=payload.get("sub",""),
          ip=request.client.host if request.client else "")

    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie("ns_token")
    return response

@router.get("/me")
async def me(payload: dict = Depends(get_current_user)):
    """Return current user profile."""
    users = _load_users()
    udata = users.get(payload["sub"], {})
    return {
        "username"    : payload["sub"],
        "role"        : payload["role"],
        "hospital_id" : payload["hospital_id"],
        "display_name": payload["display_name"],
        "email"       : payload["email"],
        "scan_count"  : udata.get("scan_count", 0),
        "last_login"  : udata.get("last_login"),
    }

@router.get("/sessions")
async def list_sessions(payload: dict = Depends(get_current_user)):
    """List active sessions for current user."""
    sessions = _load_sessions()
    user_sessions = {
        jti: {k: v for k, v in sess.items() if k != "username"}
        for jti, sess in sessions.items()
        if sess.get("username") == payload["sub"]
    }
    return {"sessions": user_sessions, "count": len(user_sessions)}

@router.delete("/sessions/{jti}")
async def revoke_session(jti: str, payload: dict = Depends(get_current_user)):
    """Revoke a specific session."""
    sessions = _load_sessions()
    if jti not in sessions:
        raise HTTPException(404, "Session not found")
    sess = sessions[jti]
    # Can only revoke own sessions unless admin
    if (sess.get("username") != payload["sub"]
            and payload["role"] not in ("admin","superadmin")):
        raise HTTPException(403, "Cannot revoke another user's session")
    del sessions[jti]
    _save_sessions(sessions)
    audit("session_revoked", user=payload["sub"], detail=jti)
    return {"status": "revoked"}

# ── User management ────────────────────────────────────────────────────────────
class RoleAssignment(BaseModel):
    username   : str
    role       : str
    hospital_id: Optional[str] = None

@router.post("/users/approve")
async def approve_user(body: RoleAssignment,
                        payload: dict = Depends(get_current_user)):
    """
    Hospital admin approves a pending user and assigns their role.
    Superadmin can approve anyone.
    Hospital admin can only approve users in their hospital.
    """
    if payload["role"] not in ("admin","superadmin"):
        raise HTTPException(403, "Admin role required")

    users = _load_users()
    if body.username not in users:
        raise HTTPException(404, "User not found")

    user = users[body.username]

    # Hospital admin scoping
    if payload["role"] == "admin":
        if user.get("hospital_id") != payload["hospital_id"]:
            raise HTTPException(403, "Cannot approve users outside your hospital")
        if body.role in ("admin","superadmin"):
            raise HTTPException(403, "Hospital admin cannot assign admin/superadmin roles")

    valid_roles = ["clinician","viewer","admin","superadmin"]
    if body.role not in valid_roles:
        raise HTTPException(400, f"Invalid role. Must be one of: {valid_roles}")

    users[body.username]["role"]       = body.role
    users[body.username]["approved"]   = True
    users[body.username]["hospital_id"]= body.hospital_id or user.get("hospital_id")
    users[body.username]["approved_by"]= payload["sub"]
    users[body.username]["approved_at"]= datetime.now().isoformat()
    _save_users(users)

    audit("user_approved", user=payload["sub"],
          detail=f"{body.username} -> {body.role}",
          hospital=payload.get("hospital_id",""))
    return {"status": "approved", "username": body.username, "role": body.role}

@router.get("/users")
async def list_users(payload: dict = Depends(get_current_user)):
    """
    List users.
    Superadmin sees all. Hospital admin sees only their hospital.
    """
    if payload["role"] not in ("admin","superadmin"):
        raise HTTPException(403, "Admin required")

    users  = _load_users()
    result = []
    for uname, udata in users.items():
        # Hospital admin scope
        if (payload["role"] == "admin"
                and udata.get("hospital_id") != payload["hospital_id"]):
            continue
        result.append({
            "username"    : uname,
            "email"       : udata.get("email",""),
            "role"        : udata.get("role",""),
            "hospital_id" : udata.get("hospital_id"),
            "active"      : udata.get("active", True),
            "approved"    : udata.get("approved", False),
            "last_login"  : udata.get("last_login"),
            "scan_count"  : udata.get("scan_count", 0),
            "display_name": udata.get("display_name",""),
        })
    return {"users": result, "count": len(result)}

@router.delete("/users/{username}")
async def deactivate_user(username: str,
                           payload: dict = Depends(get_current_user)):
    """Deactivate a user (does not delete, just sets active=False)."""
    if payload["role"] not in ("admin","superadmin"):
        raise HTTPException(403, "Admin required")

    users = _load_users()
    if username not in users:
        raise HTTPException(404, "User not found")
    if username == payload["sub"]:
        raise HTTPException(400, "Cannot deactivate yourself")

    # Hospital admin scope
    if (payload["role"] == "admin"
            and users[username].get("hospital_id") != payload["hospital_id"]):
        raise HTTPException(403, "Cannot deactivate users outside your hospital")

    users[username]["active"]         = False
    users[username]["deactivated_by"] = payload["sub"]
    users[username]["deactivated_at"] = datetime.now().isoformat()
    _save_users(users)

    # Revoke all active sessions for this user
    sessions = _load_sessions()
    to_del   = [jti for jti, s in sessions.items() if s.get("username")==username]
    for jti in to_del:
        del sessions[jti]
    if to_del:
        _save_sessions(sessions)

    audit("user_deactivated", user=payload["sub"], detail=username,
          hospital=payload.get("hospital_id",""))
    return {"status": "deactivated", "username": username}

# ── Hospital management ────────────────────────────────────────────────────────
class HospitalRegistration(BaseModel):
    name          : str
    country       : str
    hospital_type : str
    contact_email : str
    tenant_id     : Optional[str] = None
    email_domains : list          = []
    intended_use  : str           = "research"
    expected_users: int           = 10

@router.post("/hospitals/register")
@limiter.limit("5/hour")
async def register_hospital(request: Request, body: HospitalRegistration):
    """Public endpoint — any hospital can submit a registration request."""
    hospitals = _load_hospitals()
    pending   = _load_pending()

    reg_id = f"reg_{uuid.uuid4().hex[:8]}"
    pending[reg_id] = {
        "status"        : "pending",
        "name"          : body.name,
        "country"       : body.country,
        "hospital_type" : body.hospital_type,
        "contact_email" : body.contact_email,
        "tenant_id"     : body.tenant_id,
        "email_domains" : body.email_domains,
        "intended_use"  : body.intended_use,
        "expected_users": body.expected_users,
        "submitted_at"  : datetime.now().isoformat(),
        "ip"            : request.client.host if request.client else "",
    }
    _save_pending(pending)

    audit("hospital_registration_submitted", detail=body.name,
          ip=request.client.host if request.client else "")
    return {
        "status"        : "submitted",
        "registration_id": reg_id,
        "message"       : "Registration submitted. Superadmin will review within 48 hours.",
    }

@router.post("/hospitals/{reg_id}/approve")
async def approve_hospital(reg_id: str,
                            payload: dict = Depends(require_role("superadmin"))):
    """Superadmin approves a hospital registration."""
    pending = _load_pending()
    if reg_id not in pending:
        raise HTTPException(404, "Registration not found")

    reg        = pending[reg_id]
    hospitals  = _load_hospitals()
    h_id       = f"hospital_{uuid.uuid4().hex[:8]}"
    hospitals[h_id] = {
        "name"         : reg["name"],
        "country"      : reg["country"],
        "contact_email": reg["contact_email"],
        "tenant_ids"   : [reg["tenant_id"]] if reg.get("tenant_id") else [],
        "email_domains": reg.get("email_domains", []),
        "active"       : True,
        "created_at"   : datetime.now().isoformat(),
        "approved_by"  : payload["sub"],
        "scan_count"   : 0,
    }
    _save_json(HOSPITALS_FILE, hospitals)

    pending[reg_id]["status"]      = "approved"
    pending[reg_id]["approved_at"] = datetime.now().isoformat()
    pending[reg_id]["hospital_id"] = h_id
    _save_pending(pending)

    audit("hospital_approved", user=payload["sub"],
          detail=f"{reg['name']} -> {h_id}")
    return {"status": "approved", "hospital_id": h_id, "name": reg["name"]}

@router.get("/hospitals")
async def list_hospitals(payload: dict = Depends(get_current_user)):
    """
    Superadmin sees all hospitals.
    Everyone else sees only their hospital.
    """
    hospitals = _load_hospitals()
    if payload["role"] == "superadmin":
        return {"hospitals": hospitals, "count": len(hospitals)}
    # Scoped to own hospital
    h_id = payload.get("hospital_id")
    if h_id and h_id in hospitals:
        return {"hospitals": {h_id: hospitals[h_id]}, "count": 1}
    return {"hospitals": {}, "count": 0}

# ── Audit log ──────────────────────────────────────────────────────────────────
@router.get("/audit")
async def get_audit_log(limit: int = 100,
                         payload: dict = Depends(require_role("admin","superadmin"))):
    """Return recent audit log entries."""
    entries = []
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines[-limit:]):
            try:
                entry = json.loads(line.strip())
                # Hospital admin: filter to own hospital only
                if (payload["role"] == "admin"
                        and entry.get("hospital") != payload["hospital_id"]
                        and entry.get("hospital") != ""):
                    continue
                entries.append(entry)
            except:
                pass
    return {"entries": entries, "count": len(entries)}

# ── Stats endpoint ─────────────────────────────────────────────────────────────
@router.get("/stats")
async def auth_stats(payload: dict = Depends(get_current_user)):
    """Active sessions, pending users, etc."""
    sessions = _load_sessions()
    users    = _load_users()
    pending  = [u for u in users.values() if not u.get("approved", False)]
    active   = len(sessions)

    if payload["role"] == "superadmin":
        return {
            "active_sessions" : active,
            "pending_approvals": len(pending),
            "total_users"     : len(users),
            "hospitals"       : len(_load_hospitals()),
        }
    # Hospital admin — scoped stats
    h_id       = payload.get("hospital_id")
    h_users    = [u for u in users.values() if u.get("hospital_id")==h_id]
    h_pending  = [u for u in h_users if not u.get("approved", False)]
    h_sessions = [s for s in sessions.values() if s.get("hospital_id")==h_id]
    return {
        "active_sessions"  : len(h_sessions),
        "pending_approvals": len(h_pending),
        "total_users"      : len(h_users),
    }

# ── Token refresh ──────────────────────────────────────────────────────────────
@router.post("/refresh")
async def refresh_token(request: Request,
                         payload: dict = Depends(get_current_user)):
    """
    Issue a new JWT if the current one is valid and session is active.
    Called by the UI before token expiry.
    """
    new_token = create_jwt(
        payload["sub"], payload["role"],
        payload["hospital_id"], payload["display_name"],
        payload["email"]
    )
    new_payload = decode_jwt(new_token)
    sessions    = _load_sessions()
    old_jti     = payload.get("jti","")

    # Transfer session data to new JTI
    old_sess = sessions.pop(old_jti, {})
    sessions[new_payload["jti"]] = {
        **old_sess,
        "last_activity": datetime.now().isoformat(),
    }
    _save_sessions(sessions)
    audit("token_refreshed", user=payload["sub"])
    return {"token": new_token, "role": payload["role"]}

# ── Init on import ─────────────────────────────────────────────────────────────
_init_defaults()
