"""
NeuroScope AI - JWT Authentication
====================================
Provides token-based authentication for all API endpoints.

Usage:
  1. Set SECRET_KEY in environment or .env file
  2. Create users via /auth/register (admin only in production)
  3. Get token via /auth/token
  4. Pass token as: Authorization: Bearer <token>

Roles:
  admin      -- full access, user management
  clinician  -- analyze, reports, all read
  viewer     -- read-only (reports, status)
  api_client -- programmatic access (no UI)
"""

import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

try:
    from jose import JWTError, jwt
    JOSE_AVAILABLE = True
except ImportError:
    JOSE_AVAILABLE = False
    print('WARNING: python-jose not installed. Run: pip install python-jose[cryptography]')
    print('JWT auth will be DISABLED until installed.')

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY      = os.environ.get('NEUROSCOPE_SECRET_KEY', secrets.token_hex(32))
ALGORITHM       = 'HS256'
TOKEN_EXPIRE_MINS = int(os.environ.get('TOKEN_EXPIRE_MINS', 480))   # 8 hours

BASE_PATH    = os.environ.get(
    'NEUROSCOPE_BASE',
    r'C:\Users\tejan\OneDrive\Desktop\drive\NeuroScope_AI'
)
USERS_FILE   = os.path.join(BASE_PATH, 'configs', 'users.json')
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/auth/token', auto_error=False)

router = APIRouter(prefix='/auth', tags=['authentication'])


# ── Pydantic models ───────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token : str
    token_type   : str
    expires_in   : int
    role         : str
    username     : str


class TokenData(BaseModel):
    username : Optional[str] = None
    role     : Optional[str] = None


class UserCreate(BaseModel):
    username : str
    password : str
    role     : str = 'viewer'
    full_name: str = ''
    email    : str = ''


class UserOut(BaseModel):
    username  : str
    role      : str
    full_name : str
    email     : str
    created_at: str
    active    : bool


# ── Password hashing (no bcrypt dependency) ───────────────────────────────────
def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """SHA-256 with salt. Use bcrypt in production for stronger security."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f'{salt}{password}'.encode()).hexdigest()
    return hashed, salt


def verify_password(plain: str, hashed: str, salt: str) -> bool:
    computed, _ = hash_password(plain, salt)
    return computed == hashed


# ── User store ────────────────────────────────────────────────────────────────
def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_users(users: dict):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def get_user(username: str) -> Optional[dict]:
    return load_users().get(username)


def create_user(username: str, password: str, role: str = 'viewer',
                full_name: str = '', email: str = '') -> dict:
    users = load_users()
    if username in users:
        raise ValueError(f'User {username} already exists')

    hashed, salt = hash_password(password)
    user = {
        'username'  : username,
        'hashed_pw' : hashed,
        'salt'      : salt,
        'role'      : role,
        'full_name' : full_name,
        'email'     : email,
        'created_at': datetime.now().isoformat(),
        'active'    : True,
    }
    users[username] = user
    save_users(users)
    return user


def bootstrap_default_users():
    """Create default users on first run if none exist."""
    users = load_users()
    if not users:
        # Admin user
        create_user('admin',     'neuroscope_admin_2024',  'admin',
                    'NeuroScope Admin', 'admin@neuroscope.ai')
        # Clinician demo user
        create_user('clinician', 'neuroscope_clinic_2024', 'clinician',
                    'Demo Clinician', 'clinician@neuroscope.ai')
        # Viewer demo user
        create_user('viewer',    'neuroscope_view_2024',   'viewer',
                    'Demo Viewer', 'viewer@neuroscope.ai')
        print('Default users created:')
        print('  admin     / neuroscope_admin_2024  (admin)')
        print('  clinician / neuroscope_clinic_2024 (clinician)')
        print('  viewer    / neuroscope_view_2024   (viewer)')
        print('IMPORTANT: Change these passwords before production deployment!')


# ── JWT token operations ──────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    if not JOSE_AVAILABLE:
        return 'jwt-disabled'
    to_encode = data.copy()
    expire    = datetime.utcnow() + (expires_delta or timedelta(minutes=TOKEN_EXPIRE_MINS))
    to_encode.update({'exp': expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenData]:
    if not JOSE_AVAILABLE:
        return TokenData(username='anonymous', role='admin')   # permissive if no jose
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get('sub')
        role     = payload.get('role', 'viewer')
        if username is None:
            return None
        return TokenData(username=username, role=role)
    except JWTError:
        return None


# ── Dependency: get current user ──────────────────────────────────────────────
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """
    Extracts and validates JWT from Authorization header.
    Returns None if no token (unauthenticated).
    Raises 401 if token is invalid.
    """
    if not token:
        return None   # no token -- caller decides if required

    token_data = decode_token(token)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid or expired token',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    user = get_user(token_data.username)
    if not user or not user.get('active'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User not found or inactive',
        )
    return user


async def require_auth(user: dict = Depends(get_current_user)) -> dict:
    """Dependency that requires authentication (raises 401 if not logged in)."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authentication required',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    return user


async def require_clinician(user: dict = Depends(require_auth)) -> dict:
    """Requires clinician or admin role."""
    if user['role'] not in ('admin', 'clinician', 'api_client'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Role {user["role"]} does not have access to this endpoint',
        )
    return user


async def require_admin(user: dict = Depends(require_auth)) -> dict:
    """Requires admin role."""
    if user['role'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Admin role required',
        )
    return user


# ── Auth routes ───────────────────────────────────────────────────────────────
@router.post('/token', response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Get JWT access token.
    Use with: Authorization: Bearer <token>
    """
    user = get_user(form.username)
    if not user or not verify_password(form.password, user['hashed_pw'], user['salt']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect username or password',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    if not user.get('active', True):
        raise HTTPException(status_code=400, detail='Account disabled')

    token = create_access_token(
        data={'sub': user['username'], 'role': user['role']},
        expires_delta=timedelta(minutes=TOKEN_EXPIRE_MINS),
    )
    return Token(
        access_token=token,
        token_type='bearer',
        expires_in=TOKEN_EXPIRE_MINS * 60,
        role=user['role'],
        username=user['username'],
    )


@router.get('/me', response_model=UserOut)
async def get_me(current_user: dict = Depends(require_auth)):
    """Get current user profile."""
    return UserOut(
        username  = current_user['username'],
        role      = current_user['role'],
        full_name = current_user.get('full_name', ''),
        email     = current_user.get('email', ''),
        created_at= current_user.get('created_at', ''),
        active    = current_user.get('active', True),
    )


@router.post('/register', response_model=UserOut)
async def register_user(
    new_user: UserCreate,
    admin: dict = Depends(require_admin),
):
    """
    Create a new user. Admin only.
    Roles: admin | clinician | viewer | api_client
    """
    valid_roles = ('admin', 'clinician', 'viewer', 'api_client')
    if new_user.role not in valid_roles:
        raise HTTPException(400, f'Invalid role. Must be one of: {valid_roles}')

    try:
        user = create_user(
            new_user.username, new_user.password, new_user.role,
            new_user.full_name, new_user.email
        )
        return UserOut(
            username=user['username'], role=user['role'],
            full_name=user['full_name'], email=user['email'],
            created_at=user['created_at'], active=user['active'],
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get('/users', response_model=List[UserOut])
async def list_users(admin: dict = Depends(require_admin)):
    """List all users. Admin only."""
    users = load_users()
    return [
        UserOut(
            username=u['username'], role=u['role'],
            full_name=u.get('full_name',''), email=u.get('email',''),
            created_at=u.get('created_at',''), active=u.get('active', True),
        )
        for u in users.values()
    ]


@router.delete('/users/{username}')
async def deactivate_user(username: str, admin: dict = Depends(require_admin)):
    """Deactivate a user. Admin only."""
    if username == admin['username']:
        raise HTTPException(400, 'Cannot deactivate yourself')
    users = load_users()
    if username not in users:
        raise HTTPException(404, f'User {username} not found')
    users[username]['active'] = False
    save_users(users)
    return {'message': f'User {username} deactivated'}


@router.post('/change-password')
async def change_password(
    old_password: str,
    new_password: str,
    current_user: dict = Depends(require_auth),
):
    """Change own password."""
    if not verify_password(old_password, current_user['hashed_pw'], current_user['salt']):
        raise HTTPException(400, 'Incorrect current password')
    if len(new_password) < 8:
        raise HTTPException(400, 'Password must be at least 8 characters')

    users = load_users()
    hashed, salt = hash_password(new_password)
    users[current_user['username']]['hashed_pw'] = hashed
    users[current_user['username']]['salt']       = salt
    save_users(users)
    return {'message': 'Password changed successfully'}
