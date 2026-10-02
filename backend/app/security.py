"""Password hashing and signed bearer-token helpers."""
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from jose import JWTError, jwt
from .config import SECRET_KEY

ALGORITHM = "HS256"
TOKEN_TTL_HOURS = 24


def hash_password(password: str) -> str:
    """Hash a password using the same salted scrypt format as seeded accounts."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    """Verify the salted scrypt password format used by the data seeder."""
    if encoded.startswith("scrypt$"):
        try:
            _, salt_hex, digest_hex = encoded.split("$", 2)
            digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (ValueError, TypeError):
            return False
    return False


def create_access_token(user_id: int) -> str:
    """Create a signed JWT with a 24-hour expiry."""
    now = datetime.now(timezone.utc)
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + timedelta(hours=TOKEN_TTL_HOURS)},
                      SECRET_KEY, algorithm=ALGORITHM)


def token_user_id(token: str) -> int | None:
    """Decode a bearer token, returning no identity when invalid or expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, TypeError, ValueError):
        return None
