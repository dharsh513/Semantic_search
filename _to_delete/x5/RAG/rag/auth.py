"""
Authentication — accounts, password hashing and server-side sessions.

Design notes
------------
* Passwords are hashed with **PBKDF2-HMAC-SHA256**, 260,000 iterations, a fresh
  16-byte salt per user. This is in the Python standard library, so the project
  gains no new dependency and no native build step on Windows.
* The stored format is ``pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>``,
  which carries its own parameters — raising the iteration count later still
  verifies old hashes, and they are transparently upgraded on next login.
* Verification uses :func:`hmac.compare_digest` so a wrong password takes the
  same time to reject regardless of how much of it was right.
* Sessions are **server-side**. The cookie holds nothing but a 256-bit random
  token; the user id, expiry and user-agent live in SQLite. Logging out deletes
  the row, so a stolen cookie stops working immediately — unlike a signed
  cookie, which stays valid until it expires.
* Failed logins are rate limited per email+IP with escalating lockout, which
  makes online password guessing impractical without touching the database.

This is deliberately self-contained rather than Flask-Login + passlib: fewer
moving parts to install, and every security decision is visible in one file.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from flask import g, jsonify, redirect, request, url_for

from config import config
from rag.store import store

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #
ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 260_000
SALT_BYTES = 16


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "{}${}${}${}".format(
        ALGORITHM,
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> Tuple[bool, bool]:
    """
    Check a password against a stored hash.

    Returns ``(is_valid, needs_rehash)`` — the second flag is True when the hash
    was made with fewer iterations than we now use, so the caller can silently
    strengthen it.
    """
    try:
        algorithm, iters, salt_b64, hash_b64 = stored.split("$", 3)
        if algorithm != ALGORITHM:
            return False, False
        iterations = int(iters)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError, base64.binascii.Error):  # type: ignore[attr-defined]
        return False, False

    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    ok = hmac.compare_digest(candidate, expected)
    return ok, ok and iterations < ITERATIONS


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
MIN_PASSWORD = 8
MAX_PASSWORD = 200

# Rejected outright — these are the passwords credential-stuffing tries first.
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789",
    "1234567890", "qwerty123", "qwertyuiop", "letmein1", "welcome1",
    "abc12345", "iloveyou", "admin123", "root1234", "changeme",
    "passw0rd", "p@ssword", "football", "baseball", "sunshine",
    "princess", "trustno1", "starwars", "monkey12", "dragon123",
    "pubmed123", "research", "12341234", "11111111", "00000000",
}


class AuthError(Exception):
    """Raised for any auth failure. `field` lets the UI highlight the input."""

    def __init__(self, message: str, field: str = "", status: int = 400):
        super().__init__(message)
        self.message = message
        self.field = field
        self.status = status


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not email:
        raise AuthError("Enter your email address.", "email")
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise AuthError("That does not look like a valid email address.", "email")
    return email


def validate_name(name: str) -> str:
    name = re.sub(r"\s+", " ", (name or "")).strip()
    if len(name) < 2:
        raise AuthError("Enter your name (at least 2 characters).", "name")
    if len(name) > 80:
        raise AuthError("That name is too long (80 characters maximum).", "name")
    return name


def password_problems(password: str, email: str = "", name: str = "") -> Optional[str]:
    """Return the first reason a password is unacceptable, or None if it is fine."""
    password = password or ""
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters."
    if len(password) > MAX_PASSWORD:
        return f"Passwords cannot exceed {MAX_PASSWORD} characters."
    lowered = password.lower()
    if lowered in COMMON_PASSWORDS:
        return "That password is too common — pick something less guessable."
    if password.isdigit():
        return "Use more than just numbers."
    if len(set(password)) < 4:
        return "Use a wider mix of characters."
    local = (email or "").split("@")[0].lower()
    if local and len(local) > 2 and local in lowered:
        return "Do not use your email address in your password."
    if name and len(name) > 2 and name.lower().replace(" ", "") in lowered:
        return "Do not use your name in your password."
    return None


def password_strength(password: str) -> Dict[str, Any]:
    """A 0-4 score plus a label, mirrored by the meter in the browser."""
    password = password or ""
    score = 0
    if len(password) >= MIN_PASSWORD:
        score += 1
    if len(password) >= 12:
        score += 1
    classes = sum(
        bool(re.search(pattern, password))
        for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]")
    )
    if classes >= 2:
        score += 1
    if classes >= 3 and len(password) >= 10:
        score += 1
    if password.lower() in COMMON_PASSWORDS:
        score = 0
    labels = ["Very weak", "Weak", "Fair", "Good", "Strong"]
    return {"score": score, "label": labels[min(score, 4)]}


# --------------------------------------------------------------------------- #
# Login throttling
# --------------------------------------------------------------------------- #
class _Throttle:
    """In-memory failed-login counter with an escalating lockout."""

    def __init__(self, threshold: int = 5, base_lockout: int = 30,
                 max_lockout: int = 900, window: int = 900):
        self.threshold = threshold
        self.base_lockout = base_lockout
        self.max_lockout = max_lockout
        self.window = window
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, float]] = {}

    def _prune(self, now: float) -> None:
        stale = [k for k, v in self._state.items() if now - v["last"] > self.window]
        for key in stale:
            self._state.pop(key, None)

    def retry_after(self, key: str) -> int:
        with self._lock:
            now = time.time()
            self._prune(now)
            entry = self._state.get(key)
            if not entry:
                return 0
            remaining = entry.get("until", 0) - now
            return int(remaining) + 1 if remaining > 0 else 0

    def record_failure(self, key: str) -> int:
        with self._lock:
            now = time.time()
            self._prune(now)
            entry = self._state.setdefault(key, {"count": 0, "last": now, "until": 0})
            entry["count"] += 1
            entry["last"] = now
            if entry["count"] >= self.threshold:
                over = entry["count"] - self.threshold
                lockout = min(self.base_lockout * (2 ** over), self.max_lockout)
                entry["until"] = now + lockout
                return int(lockout)
            return 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._state.pop(key, None)


throttle = _Throttle()


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


# --------------------------------------------------------------------------- #
# Account operations
# --------------------------------------------------------------------------- #
def signup(email: str, name: str, password: str) -> Dict[str, Any]:
    email = validate_email(email)
    name = validate_name(name)

    problem = password_problems(password, email, name)
    if problem:
        raise AuthError(problem, "password")

    if store.user_by_email(email):
        raise AuthError(
            "An account with that email already exists. Try signing in instead.",
            "email", 409,
        )

    first_account = store.count_users() == 0
    user_id = store.create_user(email, name, hash_password(password))

    adopted = {}
    if first_account:
        # A database that already had history from before accounts existed
        # hands it to whoever signs up first, rather than orphaning it.
        adopted = store.adopt_orphan_data(user_id)
        if any(adopted.values()):
            log.info("Adopted pre-accounts data into user %s: %s", user_id, adopted)

    user = store.user_by_id(user_id)
    return {"user": user, "first_account": first_account, "adopted": adopted}


def login(email: str, password: str) -> Dict[str, Any]:
    email = validate_email(email)
    if not password:
        raise AuthError("Enter your password.", "password")

    key = f"{email}|{_client_ip()}"
    wait = throttle.retry_after(key)
    if wait:
        raise AuthError(
            f"Too many failed attempts. Try again in {wait} second(s).",
            "password", 429,
        )

    record = store.user_by_email(email, with_hash=True)

    # Always run a hash comparison, even for an unknown email, so response time
    # does not reveal whether the account exists.
    stored_hash = record["password_hash"] if record else hash_password("decoy", 1000)
    ok, needs_rehash = verify_password(password, stored_hash)

    if not record or not ok:
        lockout = throttle.record_failure(key)
        message = "Email or password is incorrect."
        if lockout:
            message += f" Too many attempts — locked for {lockout} second(s)."
        raise AuthError(message, "password", 401)

    if not record.get("is_active", 1):
        raise AuthError("This account has been deactivated.", "email", 403)

    throttle.reset(key)

    if needs_rehash:
        store.set_password_hash(record["id"], hash_password(password))

    store.touch_login(record["id"])
    return {"user": store.user_by_id(record["id"])}


def change_password(user_id: int, current: str, new: str) -> None:
    record = store.user_by_id(user_id)
    if not record:
        raise AuthError("Account not found.", "", 404)

    with_hash = store.user_by_email(record["email"], with_hash=True)
    ok, _ = verify_password(current, with_hash["password_hash"])
    if not ok:
        raise AuthError("Your current password is incorrect.", "current_password", 401)

    problem = password_problems(new, record["email"], record.get("name", ""))
    if problem:
        raise AuthError(problem, "new_password")

    store.set_password_hash(user_id, hash_password(new))
    store.delete_user_sessions(user_id)   # force a fresh login everywhere


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def start_session(user_id: int, remember: bool = False) -> Tuple[str, int]:
    """Create a session and return ``(token, max_age_seconds)``."""
    token = secrets.token_urlsafe(32)
    days = config.SESSION_REMEMBER_DAYS if remember else config.SESSION_DAYS
    max_age = int(days * 86400)
    store.create_session(
        token, user_id, time.time() + max_age,
        request.headers.get("User-Agent", "") if request else "",
    )
    # Opportunistic housekeeping; cheap and keeps the table from growing.
    if secrets.randbelow(20) == 0:
        store.purge_expired_sessions()
    return token, max_age


def end_session(token: str) -> bool:
    return store.delete_session(token) if token else False


def current_user() -> Optional[Dict[str, Any]]:
    """The signed-in user for this request, resolved once and cached on `g`."""
    if not hasattr(g, "_current_user"):
        token = request.cookies.get(config.SESSION_COOKIE)
        g._current_user = store.session_user(token) if token else None
    return g._current_user


def set_session_cookie(response, token: str, max_age: int):
    response.set_cookie(
        config.SESSION_COOKIE,
        token,
        max_age=max_age,
        httponly=True,                       # unreachable from JavaScript
        samesite="Lax",                      # blocks cross-site form CSRF
        secure=bool(config.SESSION_COOKIE_SECURE),
        path="/",
    )
    return response


def clear_session_cookie(response):
    response.delete_cookie(config.SESSION_COOKIE, path="/")
    return response


# --------------------------------------------------------------------------- #
# Route protection
# --------------------------------------------------------------------------- #
def _wants_json() -> bool:
    if request.path.startswith("/api/"):
        return True
    accept = request.headers.get("Accept", "")
    return "application/json" in accept and "text/html" not in accept


def login_required(view: Callable) -> Callable:
    """
    Reject anonymous requests.

    API calls get a 401 with a JSON body so the frontend can react; page
    requests are redirected to the auth screen with a `next` parameter so the
    user lands back where they were trying to go.
    """

    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            if _wants_json():
                return jsonify({
                    "error": "Authentication required.",
                    "auth_required": True,
                }), 401
            return redirect(url_for("auth_page", next=request.full_path.rstrip("?")))
        g.user = user
        return view(*args, **kwargs)

    return wrapper


def current_user_id() -> int:
    """The signed-in user's id — 0 when nobody is signed in."""
    user = getattr(g, "user", None) or current_user()
    return int(user["id"]) if user else 0
