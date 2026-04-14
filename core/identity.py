"""
core/identity.py — MESIM device identity and rank management

Each field device has a persistent identity comprising:
  - UUID4 device_id (stable across reboots)
  - Callsign (e.g., "ALPHA-1"), max 32 chars, [A-Z0-9_-]
  - Rank (COMMAND > OFFICER > NCO > SQUAD)
  - Ed25519 signing keypair (authenticate messages)
  - X25519 encryption keypair (key exchange)
  - ML-KEM-768 KEM keypair (post-quantum key exchange)

Identities are persisted to disk with private keys encrypted via:
  Argon2id (or Scrypt fallback) KDF → ChaCha20-Poly1305 key-wrap

The PublicBundle (public keys + metadata + signature) is what nodes
broadcast during mesh discovery.
"""

from __future__ import annotations

import base64
import json
import os
import re
import struct
import uuid
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from core.crypto import (
    Ed25519KeyPair,
    Ed25519SigningKey,
    Ed25519VerifyKey,
    MLKEMKeyPair,
    MLKEMPublicKey,
    MLKEMSecretKey,
    SessionKey,
    X25519KeyPair,
    X25519PrivKey,
    X25519PubKey,
    _MLKEM_PK_LEN,
    _MLKEM_SK_LEN,
    generate_ed25519_keypair,
    generate_mlkem_keypair,
    generate_x25519_keypair,
    sign_message,
    verify_signature,
)

_IDENTITY_FILE_VERSION = 1
_CALLSIGN_RE = re.compile(r"^[A-Z0-9_-]{1,32}$")

# KDF parameters
_ARGON2_TIME_COST = 3
_ARGON2_MEMORY_COST = 65536  # 64 MiB
_ARGON2_PARALLELISM = 4
_ARGON2_HASH_LEN = 32

try:
    import argon2.low_level as _argon2_ll
    import argon2 as _argon2_mod

    _argon2_available = True
except ImportError:
    _argon2_available = False


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


class Rank(IntEnum):
    """
    Operational rank. Lower integer = higher authority.
    Used for ACL enforcement in the mesh router.
    """

    COMMAND = 1
    OFFICER = 2
    NCO = 3
    SQUAD = 4


# ---------------------------------------------------------------------------
# DeviceIdentity
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DeviceIdentity:
    """
    Full device identity including all keypairs.
    The private key fields must never be serialized in plaintext.
    """

    device_id: str  # UUID4 string
    callsign: str  # [A-Z0-9_-], 1–32 chars, auto-uppercased on creation
    rank: Rank
    signing_keypair: Ed25519KeyPair
    encrypt_keypair: X25519KeyPair
    kem_keypair: MLKEMKeyPair


# ---------------------------------------------------------------------------
# PublicBundle — what is broadcast to peers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PublicBundle:
    """
    Public portion of a device identity, signed by the device's signing key.

    Peers verify bundle_sig before trusting any field. The signature covers
    ALL other fields in canonical order, preventing substitution attacks.
    """

    device_id: str
    callsign: str
    rank: Rank
    verify_key: Ed25519VerifyKey
    encrypt_pub: X25519PubKey
    kem_pub: MLKEMPublicKey
    bundle_sig: bytes  # Ed25519 signature over _canonical_bundle_bytes(...)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_passphrase(passphrase: str | bytes) -> bytes:
    if isinstance(passphrase, str):
        return passphrase.encode("utf-8")
    if isinstance(passphrase, bytes):
        return passphrase
    raise TypeError(f"passphrase must be str or bytes, got {type(passphrase).__name__}")


def _canonical_bundle_bytes(
    device_id: str,
    callsign: str,
    rank: Rank,
    verify_key: Ed25519VerifyKey,
    encrypt_pub: X25519PubKey,
    kem_pub: MLKEMPublicKey,
) -> bytes:
    """
    Stable, deterministic serialization of all public bundle fields for signing.
    Uses fixed-length binary packing to prevent length-extension or injection.

    Format:
      [device_id: 36 bytes UTF-8, zero-padded]
      [callsign: 32 bytes UTF-8, zero-padded]
      [rank: 1 byte big-endian]
      [verify_key: 32 bytes]
      [encrypt_pub: 32 bytes]
      [kem_pub: 1184 bytes]
    Total: 1317 bytes
    """
    device_id_b = device_id.encode("utf-8").ljust(36, b"\x00")[:36]
    callsign_b = callsign.encode("utf-8").ljust(32, b"\x00")[:32]
    rank_b = struct.pack(">B", int(rank))
    return device_id_b + callsign_b + rank_b + verify_key.raw + encrypt_pub.raw + kem_pub.raw


def _derive_wrap_key(passphrase: bytes, salt: bytes) -> bytes:
    """
    Derive a 32-byte key for wrapping (encrypting) private keys.

    Primary: Argon2id via argon2-cffi.
    Fallback: Scrypt via cryptography (if argon2-cffi not available).
    The identity file records which KDF was used.
    """
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

        kdf = Scrypt(salt=salt, length=32, n=2**17, r=8, p=1)
        return kdf.derive(passphrase)


def _kdf_name() -> str:
    return "argon2id" if _argon2_available else "scrypt"


def _kdf_params() -> dict:
    if _argon2_available:
        return {
            "time_cost": _ARGON2_TIME_COST,
            "memory_cost": _ARGON2_MEMORY_COST,
            "parallelism": _ARGON2_PARALLELISM,
            "hash_len": _ARGON2_HASH_LEN,
        }
    return {"n": 2**17, "r": 8, "p": 1}


def _derive_wrap_key_from_params(passphrase: bytes, salt: bytes, kdf: str, params: dict) -> bytes:
    """Re-derive wrap key using parameters stored in the identity file."""
    if kdf == "argon2id":
        if not _argon2_available:
            raise RuntimeError("Identity was created with Argon2id but argon2-cffi is not installed")
        return _argon2_ll.hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=params["time_cost"],
            memory_cost=params["memory_cost"],
            parallelism=params["parallelism"],
            hash_len=params["hash_len"],
            type=_argon2_mod.low_level.Type.ID,
        )
    elif kdf == "scrypt":
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

        kdf_obj = Scrypt(salt=salt, length=32, n=params["n"], r=params["r"], p=params["p"])
        return kdf_obj.derive(passphrase)
    else:
        raise ValueError(f"unsupported KDF in identity file: {kdf!r}")


def _serialize_private_keys(identity: DeviceIdentity) -> bytes:
    """
    Pack private key bytes into a fixed-length binary blob for encryption.

    Layout:
      [ed25519_signing_key: 32 bytes]
      [x25519_private_key:  32 bytes]
      [mlkem_secret_key:  2400 bytes]
    Total: 2464 bytes
    """
    return (
        identity.signing_keypair.signing_key.raw
        + identity.encrypt_keypair.private_key.raw
        + identity.kem_keypair.secret_key.raw
    )


def _deserialize_private_keys(
    raw: bytes,
) -> tuple[Ed25519KeyPair, X25519KeyPair, MLKEMKeyPair]:
    """
    Unpack private keys from the fixed-layout binary blob.
    Also reconstructs public keys from the private key material for consistency.
    """
    if len(raw) != 32 + 32 + _MLKEM_SK_LEN:
        raise ValueError(f"Private key blob has unexpected length: {len(raw)}")

    offset = 0

    ed_sk_raw = raw[offset : offset + 32]
    offset += 32

    x25519_sk_raw = raw[offset : offset + 32]
    offset += 32

    mlkem_sk_raw = raw[offset : offset + _MLKEM_SK_LEN]

    # Reconstruct public keys from private key material
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    ed_priv = Ed25519PrivateKey.from_private_bytes(ed_sk_raw)
    ed_pub_raw = ed_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    x_priv = X25519PrivateKey.from_private_bytes(x25519_sk_raw)
    x_pub_raw = x_priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    ed_kp = Ed25519KeyPair(
        signing_key=Ed25519SigningKey(raw=ed_sk_raw),
        verify_key=Ed25519VerifyKey(raw=ed_pub_raw),
    )
    x_kp = X25519KeyPair(
        private_key=X25519PrivKey(raw=x25519_sk_raw),
        public_key=X25519PubKey(raw=x_pub_raw),
    )

    # ML-KEM: public key is embedded in the secret key at bytes 1152..2336
    # per FIPS 203 (dk = dk_PKE[1152] || ek[1184] || H(ek)[32] || z[32]).
    # Extract directly — no oqs call needed, no new keypair generated.
    _MLKEM_PK_OFFSET = 1152
    mlkem_pk_raw = mlkem_sk_raw[_MLKEM_PK_OFFSET : _MLKEM_PK_OFFSET + _MLKEM_PK_LEN]

    kem_kp = MLKEMKeyPair(
        secret_key=MLKEMSecretKey(raw=mlkem_sk_raw),
        public_key=MLKEMPublicKey(raw=mlkem_pk_raw),
    )

    return ed_kp, x_kp, kem_kp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_identity(callsign: str, rank: Rank) -> DeviceIdentity:
    """
    Create a new device identity with freshly generated keypairs.

    Args:
        callsign: 1–32 character identifier, [A-Z0-9_-]. Auto-uppercased.
        rank:     Operational rank (Rank.COMMAND through Rank.SQUAD).

    Raises:
        ValueError: if callsign is invalid after uppercasing.
    """
    callsign = callsign.upper()
    if not _CALLSIGN_RE.match(callsign):
        raise ValueError(
            f"Invalid callsign {callsign!r}. Must be 1–32 chars, [A-Z0-9_-] only."
        )

    return DeviceIdentity(
        device_id=str(uuid.uuid4()),
        callsign=callsign,
        rank=rank,
        signing_keypair=generate_ed25519_keypair(),
        encrypt_keypair=generate_x25519_keypair(),
        kem_keypair=generate_mlkem_keypair(),
    )


def get_public_bundle(identity: DeviceIdentity) -> PublicBundle:
    """
    Extract the public bundle from a device identity.

    The bundle is signed over all public fields in canonical order.
    Peers must verify bundle_sig before trusting any field.
    """
    canon = _canonical_bundle_bytes(
        device_id=identity.device_id,
        callsign=identity.callsign,
        rank=identity.rank,
        verify_key=identity.signing_keypair.verify_key,
        encrypt_pub=identity.encrypt_keypair.public_key,
        kem_pub=identity.kem_keypair.public_key,
    )
    sig = sign_message(canon, identity.signing_keypair.signing_key)

    return PublicBundle(
        device_id=identity.device_id,
        callsign=identity.callsign,
        rank=identity.rank,
        verify_key=identity.signing_keypair.verify_key,
        encrypt_pub=identity.encrypt_keypair.public_key,
        kem_pub=identity.kem_keypair.public_key,
        bundle_sig=sig,
    )


def verify_public_bundle(bundle: PublicBundle) -> None:
    """
    Verify a peer's PublicBundle signature.

    Raises cryptography.exceptions.InvalidSignature if the bundle has been tampered.
    """
    canon = _canonical_bundle_bytes(
        device_id=bundle.device_id,
        callsign=bundle.callsign,
        rank=bundle.rank,
        verify_key=bundle.verify_key,
        encrypt_pub=bundle.encrypt_pub,
        kem_pub=bundle.kem_pub,
    )
    verify_signature(canon, bundle.bundle_sig, bundle.verify_key)


def save_identity(
    identity: DeviceIdentity,
    path: str | os.PathLike,
    passphrase: str | bytes,
) -> None:
    """
    Encrypt and persist a device identity to disk.

    Private keys are encrypted with ChaCha20-Poly1305 using a key derived
    from the passphrase via Argon2id (or Scrypt fallback).

    Args:
        identity:   The DeviceIdentity to save.
        path:       Destination file path (.json recommended).
        passphrase: Encryption passphrase (str or bytes).

    Raises:
        OSError: on write failure.
    """
    passphrase_b = _normalize_passphrase(passphrase)
    salt = os.urandom(32)
    nonce = os.urandom(12)
    wrap_key = _derive_wrap_key(passphrase_b, salt)

    private_blob = _serialize_private_keys(identity)
    chacha = ChaCha20Poly1305(wrap_key)
    encrypted_private = chacha.encrypt(nonce, private_blob, None)

    doc = {
        "version": _IDENTITY_FILE_VERSION,
        "device_id": identity.device_id,
        "callsign": identity.callsign,
        "rank": int(identity.rank),
        "kdf": _kdf_name(),
        "kdf_params": _kdf_params(),
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "public_keys": {
            "verify_key": base64.b64encode(identity.signing_keypair.verify_key.raw).decode(),
            "encrypt_pub": base64.b64encode(identity.encrypt_keypair.public_key.raw).decode(),
            "kem_pub": base64.b64encode(identity.kem_keypair.public_key.raw).decode(),
        },
        "encrypted_private_keys": base64.b64encode(encrypted_private).decode(),
    }

    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_identity(
    path: str | os.PathLike,
    passphrase: str | bytes,
) -> DeviceIdentity:
    """
    Load and decrypt a device identity from disk.

    Args:
        path:       Path to the identity JSON file.
        passphrase: Decryption passphrase (str or bytes).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:         on version mismatch or malformed file.
        InvalidTag:         if passphrase is wrong OR file is corrupted.
                            (Deliberately indistinguishable to prevent oracles.)
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    doc = json.loads(raw_text)

    version = doc.get("version")
    if version != _IDENTITY_FILE_VERSION:
        raise ValueError(f"unsupported identity file version: {version!r}")

    passphrase_b = _normalize_passphrase(passphrase)
    salt = base64.b64decode(doc["salt"])
    nonce = base64.b64decode(doc["nonce"])
    encrypted_private = base64.b64decode(doc["encrypted_private_keys"])

    wrap_key = _derive_wrap_key_from_params(passphrase_b, salt, doc["kdf"], doc["kdf_params"])
    chacha = ChaCha20Poly1305(wrap_key)

    # InvalidTag propagates directly — same error for wrong passphrase or corruption
    private_blob = chacha.decrypt(nonce, encrypted_private, None)

    ed_kp, x_kp, kem_kp = _deserialize_private_keys(private_blob)

    # Consistency check: re-derived public keys must match stored public keys
    stored_verify = base64.b64decode(doc["public_keys"]["verify_key"])
    stored_encrypt = base64.b64decode(doc["public_keys"]["encrypt_pub"])
    stored_kem_pub = base64.b64decode(doc["public_keys"]["kem_pub"])

    if ed_kp.verify_key.raw != stored_verify:
        raise ValueError("Identity file corrupted: Ed25519 public key mismatch")
    if x_kp.public_key.raw != stored_encrypt:
        raise ValueError("Identity file corrupted: X25519 public key mismatch")

    # Consistency check: public key extracted from secret key must match stored value
    if kem_kp.public_key.raw != stored_kem_pub:
        raise ValueError("Identity file corrupted: ML-KEM-768 public key mismatch")

    return DeviceIdentity(
        device_id=doc["device_id"],
        callsign=doc["callsign"],
        rank=Rank(doc["rank"]),
        signing_keypair=ed_kp,
        encrypt_keypair=x_kp,
        kem_keypair=kem_kp,
    )
