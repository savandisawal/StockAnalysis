"""Basic HTTP authentication with bcrypt-hashed passwords.

AUTH_USERS holds comma-separated "user:bcrypt_hash" pairs — never
plaintext passwords. Generate a hash with: python -m scripts.hash_password

Fails closed: with no users configured, every request is rejected.
"""

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import settings

security = HTTPBasic()

# Verifying against this when the user is unknown keeps response timing
# uniform (no user-enumeration via latency).
_DUMMY_HASH = bcrypt.hashpw(b"__dummy_password_for_timing__", bcrypt.gensalt())


def hash_password(password: str) -> str:
    """bcrypt-hash a password for an AUTH_USERS entry."""
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode()


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], stored_hash.encode())
    except ValueError:
        return False


def is_bcrypt_hash(value: str) -> bool:
    """True if value looks like a bcrypt hash ($2a$/$2b$/$2y$ prefix)."""
    return value.startswith(("$2a$", "$2b$", "$2y$"))


def validate_auth_config() -> None:
    """Startup check: in prod, auth must be configured with bcrypt hashes.

    Raises RuntimeError so a misconfigured production deploy fails loudly
    instead of running open or with plaintext credentials.
    """
    creds = settings.get_auth_credentials()
    if settings.environment != "prod":
        return
    if not creds:
        raise RuntimeError(
            "AUTH_USERS is empty — production requires at least one "
            "user:bcrypt_hash entry (python -m scripts.hash_password)"
        )
    for user, pwd_hash in creds.items():
        if not is_bcrypt_hash(pwd_hash):
            raise RuntimeError(
                f"AUTH_USERS entry for '{user}' is not a bcrypt hash — "
                "plaintext passwords are not allowed in production"
            )


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """FastAPI dependency — validates username:password against bcrypt hashes.

    Returns the authenticated username.
    """
    valid_users = settings.get_auth_credentials()

    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )

    stored_hash = valid_users.get(credentials.username)
    if stored_hash is None or not is_bcrypt_hash(stored_hash):
        # Unknown user (or misconfigured plaintext entry): burn a bcrypt
        # verify anyway so timing doesn't reveal which usernames exist.
        bcrypt.checkpw(credentials.password.encode("utf-8")[:72], _DUMMY_HASH)
        raise unauthorized

    if not _verify_password(credentials.password, stored_hash):
        raise unauthorized

    return credentials.username
