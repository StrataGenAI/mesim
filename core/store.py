"""
core/store.py — MESIM encrypted message store

Persists messages in a regular SQLite database with application-level
ChaCha20-Poly1305 encryption on all sensitive columns.  A thin _open_db()
abstraction makes a future swap to full SQLCipher a one-line change.

Key derivation:
  Argon2id (time=3, mem=64MiB, par=4, len=32), identical parameters to
  core/identity.py.  Salt is generated once on DB creation and stored in
  the `meta` table.  A HMAC-SHA256 sentinel verifies the key on every
  re-open — wrong passphrase raises InvalidTag (indistinguishable from
  corruption, per the project oracle-prevention policy).

Schema:
  meta(key TEXT PK, value TEXT)
  messages(id INTEGER PK, peer_id TEXT, direction TEXT, ciphertext BLOB,
           timestamp REAL, delivered INTEGER)
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_lib
import json
import os
import sqlite3
import time
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# ---------------------------------------------------------------------------
# Optional Argon2id (identical params to identity.py)
# ---------------------------------------------------------------------------

try:
    import argon2.low_level as _argon2_ll
    import argon2 as _argon2_mod
    _argon2_available = True
except ImportError:
    _argon2_available = False

_ARGON2_TIME_COST   = 3
_ARGON2_MEMORY_COST = 65536   # 64 MiB
_ARGON2_PARALLELISM = 4
_ARGON2_HASH_LEN    = 32

_STORE_VERSION  = 1
_KEY_CHECK_INFO = b"mesim-store-v1"   # HMAC sentinel domain separator

_VALID_DIRECTIONS = frozenset({"sent", "recv"})


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredMessage:
    """A message record returned by get_history. plaintext is in-memory only."""

    id: int
    peer_id: str
    direction: str   # "sent" | "recv"
    plaintext: bytes
    timestamp: float
    delivered: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_passphrase(passphrase: str | bytes) -> bytes:
    if isinstance(passphrase, str):
        return passphrase.encode("utf-8")
    if isinstance(passphrase, bytes):
        return passphrase
    raise TypeError(f"passphrase must be str or bytes, got {type(passphrase).__name__}")


def _derive_key(passphrase: bytes, salt: bytes) -> bytes:
    """Derive a 32-byte store key via Argon2id (or Scrypt fallback)."""
    if _argon2_available:
        return _argon2_ll.hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=_ARGON2_TIME_COST,
            memory_cost=_ARGON2_MEMORY_COST,
            parallelism=_ARGON2_PARALLELISM,
            hash_len=_ARGON2_HASH_LEN,
            type=_argon2_mod.low_level.Type.ID,
        )
    else:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        kdf = Scrypt(salt=salt, length=32, n=2 ** 17, r=8, p=1)
        return kdf.derive(passphrase)


def _kdf_name() -> str:
    return "argon2id" if _argon2_available else "scrypt"


def _kdf_params() -> dict:
    if _argon2_available:
        return {
            "alg": "argon2id",
            "time_cost": _ARGON2_TIME_COST,
            "memory_cost": _ARGON2_MEMORY_COST,
            "parallelism": _ARGON2_PARALLELISM,
            "hash_len": _ARGON2_HASH_LEN,
        }
    return {"alg": "scrypt", "n": 2 ** 17, "r": 8, "p": 1}


def _derive_key_from_params(passphrase: bytes, salt: bytes, params: dict) -> bytes:
    alg = params.get("alg", "argon2id")
    if alg == "argon2id":
        if not _argon2_available:
            raise RuntimeError("Store was created with Argon2id but argon2-cffi is not installed")
        return _argon2_ll.hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=params["time_cost"],
            memory_cost=params["memory_cost"],
            parallelism=params["parallelism"],
            hash_len=params["hash_len"],
            type=_argon2_mod.low_level.Type.ID,
        )
    elif alg == "scrypt":
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
        kdf = Scrypt(salt=salt, length=32, n=params["n"], r=params["r"], p=params["p"])
        return kdf.derive(passphrase)
    else:
        raise ValueError(f"unsupported KDF: {alg!r}")


def _encrypt(key: bytes, plaintext: bytes) -> bytes:
    """Returns nonce (12B) || ciphertext_with_tag (len+16B)."""
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
    return nonce + ct


def _decrypt(key: bytes, blob: bytes) -> bytes:
    """Inverse of _encrypt. Raises InvalidTag on failure."""
    nonce, ct = blob[:12], blob[12:]
    return ChaCha20Poly1305(key).decrypt(nonce, ct, None)


def _key_check(key: bytes) -> bytes:
    """HMAC-SHA256 sentinel for key verification. Stored in meta on first open."""
    return hmac_lib.new(key, _KEY_CHECK_INFO, hashlib.sha256).digest()


def _open_db(path: str) -> sqlite3.Connection:
    """Open a SQLite connection. Abstraction point for future SQLCipher swap."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ---------------------------------------------------------------------------
# MessageStore
# ---------------------------------------------------------------------------


class MessageStore:
    """
    Encrypted SQLite message store for MESIM.

    Usage::

        with MessageStore("messages.db", "my-passphrase") as store:
            msg_id = store.save_message("peer_abc", "sent", b"hello")
            history = store.get_history("peer_abc")
    """

    def __init__(self, db_path: str, passphrase: str | bytes) -> None:
        self._db_path = db_path
        self._passphrase = _normalize_passphrase(passphrase)
        self._key: bytes | None = None
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        """Derive key, create/upgrade schema, verify key sentinel."""
        if self._conn is not None:
            return  # already open — idempotent

        conn = _open_db(self._db_path)
        self._conn = conn
        self._ensure_schema()

        # First open: generate salt, derive key, store sentinel
        salt_b64 = self._meta_get("kdf_salt")
        if salt_b64 is None:
            salt = os.urandom(32)
            key = _derive_key(self._passphrase, salt)
            self._meta_set("kdf_salt",   base64.b64encode(salt).decode())
            self._meta_set("kdf_params", json.dumps(_kdf_params()))
            self._meta_set("key_check",  base64.b64encode(_key_check(key)).decode())
            self._meta_set("version",    str(_STORE_VERSION))
            self._key = key
        else:
            # Re-open: re-derive key, verify sentinel
            salt = base64.b64decode(salt_b64)
            params = json.loads(self._meta_get("kdf_params") or "{}")
            key = _derive_key_from_params(self._passphrase, salt, params)
            stored_check = base64.b64decode(self._meta_get("key_check") or "")
            expected = _key_check(key)
            if not hmac_lib.compare_digest(expected, stored_check):
                self._conn.close()
                self._conn = None
                raise InvalidTag()
            self._key = key

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._key = None

    def __enter__(self) -> MessageStore:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Public API ─────────────────────────────────────────────────────────

    def save_message(
        self,
        peer_id: str,
        direction: str,
        plaintext: bytes,
        timestamp: float | None = None,
    ) -> int:
        """
        Encrypt plaintext and persist the message.

        Returns the row id. Raises ValueError for invalid direction.
        """
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be 'sent' or 'recv', got {direction!r}")
        self._assert_open()
        ts = timestamp if timestamp is not None else time.time()
        blob = _encrypt(self._key, plaintext)
        cur = self._conn.execute(
            "INSERT INTO messages (peer_id, direction, ciphertext, timestamp, delivered)"
            " VALUES (?, ?, ?, ?, 0)",
            (peer_id, direction, blob, ts),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_history(
        self,
        peer_id: str,
        limit: int = 100,
        before_id: int | None = None,
    ) -> list[StoredMessage]:
        """
        Return decrypted messages for peer_id, oldest first.

        limit: max rows (default 100).
        before_id: if given, only rows with id < before_id (pagination).
        """
        self._assert_open()
        if before_id is not None:
            cur = self._conn.execute(
                "SELECT id, peer_id, direction, ciphertext, timestamp, delivered"
                " FROM messages WHERE peer_id=? AND id<?"
                " ORDER BY timestamp ASC, id ASC LIMIT ?",
                (peer_id, before_id, limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT id, peer_id, direction, ciphertext, timestamp, delivered"
                " FROM messages WHERE peer_id=?"
                " ORDER BY timestamp ASC, id ASC LIMIT ?",
                (peer_id, limit),
            )
        rows = cur.fetchall()
        result = []
        for row_id, p_id, direction, blob, ts, delivered in rows:
            plaintext = _decrypt(self._key, bytes(blob))
            result.append(StoredMessage(
                id=row_id,
                peer_id=p_id,
                direction=direction,
                plaintext=plaintext,
                timestamp=ts,
                delivered=bool(delivered),
            ))
        return result

    def mark_delivered(self, msg_id: int) -> None:
        """Mark a message as delivered. No-op for unknown ids."""
        self._assert_open()
        self._conn.execute(
            "UPDATE messages SET delivered=1 WHERE id=?", (msg_id,)
        )
        self._conn.commit()

    def purge_old(self, max_age_seconds: float = 7 * 86400) -> int:
        """
        Delete messages older than max_age_seconds.

        Returns the number of rows deleted.
        """
        self._assert_open()
        cutoff = time.time() - max_age_seconds
        cur = self._conn.execute(
            "DELETE FROM messages WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    # ── Private helpers ────────────────────────────────────────────────────

    def _assert_open(self) -> None:
        if self._conn is None or self._key is None:
            raise RuntimeError("MessageStore is not open — call open() first")

    def _ensure_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_id     TEXT    NOT NULL,
                direction   TEXT    NOT NULL
                                CHECK (direction IN ('sent','recv')),
                ciphertext  BLOB    NOT NULL,
                timestamp   REAL    NOT NULL,
                delivered   INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_msg_peer_ts
                ON messages (peer_id, timestamp ASC);
        """)
        self._conn.commit()

    def _meta_get(self, key: str) -> str | None:
        cur = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()
