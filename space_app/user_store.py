from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Tuple


USER_DB_PATH = Path(__file__).resolve().parent / "users.json"


def _atomic_write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_username(username: str) -> str:
    return (username or "").strip()


def validate_username(username: str) -> None:
    u = _normalize_username(username)
    if not u:
        raise ValueError("用户名不能为空")
    if len(u) < 3 or len(u) > 32:
        raise ValueError("用户名长度需在 3~32")
    # 作为目录名、以及 table_id 里的分隔符：禁止 ':' '/' '\\'
    bad = set(":/\\")
    if any(ch in bad for ch in u):
        raise ValueError("用户名不能包含 : / \\")
    if not all(ch.isalnum() or ch in {"_", "-"} for ch in u):
        raise ValueError("用户名仅允许字母/数字/_/-")


def _pbkdf2(password: str, salt: bytes, rounds: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))


def _hash_password(password: str) -> Tuple[str, str, int]:
    salt = secrets.token_bytes(16)
    rounds = 200_000
    digest = _pbkdf2(password, salt, rounds=rounds)
    return base64.b64encode(salt).decode("ascii"), base64.b64encode(digest).decode("ascii"), rounds


def _verify_password(password: str, salt_b64: str, digest_b64: str, rounds: int) -> bool:
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
        digest = base64.b64decode(digest_b64.encode("ascii"))
        calc = _pbkdf2(password, salt, rounds=int(rounds))
        return hmac.compare_digest(calc, digest)
    except Exception:
        return False


def load_users() -> Dict[str, Any]:
    if not USER_DB_PATH.exists():
        return {"version": 1, "users": {}}
    try:
        doc = json.loads(USER_DB_PATH.read_text(encoding="utf-8") or "{}")
    except Exception:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    if "users" not in doc or not isinstance(doc.get("users"), dict):
        doc["users"] = {}
    doc.setdefault("version", 1)
    return doc


def save_users(doc: Dict[str, Any]) -> None:
    if not isinstance(doc, dict):
        raise ValueError("invalid users db")
    if "users" not in doc or not isinstance(doc.get("users"), dict):
        doc["users"] = {}
    doc.setdefault("version", 1)
    _atomic_write_json(USER_DB_PATH, doc)


def ensure_admin(password: str = "zblnb666", force: bool = False) -> None:
    doc = load_users()
    users = doc.get("users")
    if not isinstance(users, dict):
        users = {}
        doc["users"] = users
    if (not force) and ("admin" in users):
        return
    salt, digest, rounds = _hash_password(password)
    users["admin"] = {"salt": salt, "digest": digest, "rounds": rounds}
    save_users(doc)


def user_exists(username: str) -> bool:
    u = _normalize_username(username)
    doc = load_users()
    users = doc.get("users")
    return isinstance(users, dict) and u in users


def create_user(username: str, password: str) -> None:
    validate_username(username)
    if not password or len(password) < 6:
        raise ValueError("密码长度至少 6 位")
    u = _normalize_username(username)
    doc = load_users()
    users = doc.get("users")
    if not isinstance(users, dict):
        users = {}
        doc["users"] = users
    if u in users:
        raise ValueError("用户已存在")
    salt, digest, rounds = _hash_password(password)
    users[u] = {"salt": salt, "digest": digest, "rounds": rounds}
    save_users(doc)


def verify_user(username: str, password: str) -> bool:
    u = _normalize_username(username)
    doc = load_users()
    users = doc.get("users")
    if not isinstance(users, dict):
        return False
    meta = users.get(u)
    if not isinstance(meta, dict):
        return False
    return _verify_password(
        password,
        str(meta.get("salt") or ""),
        str(meta.get("digest") or ""),
        int(meta.get("rounds") or 0),
    )
