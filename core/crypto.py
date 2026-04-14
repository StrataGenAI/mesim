"""
core/crypto.py — MESIM cryptographic primitives

Hybrid post-quantum + classical encryption:
  - Ed25519 (signing)
  - X25519 (ECDH key exchange)
  - ML-KEM-768 (post-quantum KEM, via liboqs)
  - ChaCha20-Poly1305 (symmetric AEAD)
  - HKDF-SHA256 (key derivation)

Wire format for encrypted messages (assembled by transport layer):
  [nonce: 12B] [ChaCha20-Poly1305({ sig: 64B | plaintext: NB }, tag: 16B)]
  The signature is inside the ciphertext — sender identity hidden from observers.
"""

from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature, InvalidTag  # noqa: F401 (re-exported)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey as _X25519PublicKeyRaw,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

try:
    import oqs as _oqs

    _oqs_available = True
except ImportError:
    _oqs_available = False

_MLKEM_ALG = "ML-KEM-768"
_MLKEM_PK_LEN = 1184
_MLKEM_SK_LEN = 2400
_MLKEM_CT_LEN = 1088

# ---------------------------------------------------------------------------
# Key types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ed25519SigningKey:
    """Ed25519 private signing key (32 bytes). Never log or transmit."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise ValueError(f"Ed25519SigningKey must be 32 bytes, got {len(self.raw)}")

    def __repr__(self) -> str:
        return "Ed25519SigningKey(<REDACTED>)"

    def __str__(self) -> str:
        return "Ed25519SigningKey(<REDACTED>)"


@dataclass(frozen=True, slots=True)
class Ed25519VerifyKey:
    """Ed25519 public verify key (32 bytes)."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise ValueError(f"Ed25519VerifyKey must be 32 bytes, got {len(self.raw)}")


@dataclass(frozen=True, slots=True)
class Ed25519KeyPair:
    signing_key: Ed25519SigningKey
    verify_key: Ed25519VerifyKey


@dataclass(frozen=True, slots=True)
class X25519PrivKey:
    """X25519 private key (32 bytes). Never log or transmit."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise ValueError(f"X25519PrivKey must be 32 bytes, got {len(self.raw)}")

    def __repr__(self) -> str:
        return "X25519PrivKey(<REDACTED>)"

    def __str__(self) -> str:
        return "X25519PrivKey(<REDACTED>)"


@dataclass(frozen=True, slots=True)
class X25519PubKey:
    """X25519 public key (32 bytes)."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise ValueError(f"X25519PubKey must be 32 bytes, got {len(self.raw)}")


@dataclass(frozen=True, slots=True)
class X25519KeyPair:
    private_key: X25519PrivKey
    public_key: X25519PubKey


@dataclass(frozen=True, slots=True)
class MLKEMSecretKey:
    """ML-KEM-768 secret key (2400 bytes). Never log or transmit."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != _MLKEM_SK_LEN:
            raise ValueError(f"MLKEMSecretKey must be {_MLKEM_SK_LEN} bytes, got {len(self.raw)}")

    def __repr__(self) -> str:
        return "MLKEMSecretKey(<REDACTED>)"

    def __str__(self) -> str:
        return "MLKEMSecretKey(<REDACTED>)"


@dataclass(frozen=True, slots=True)
class MLKEMPublicKey:
    """ML-KEM-768 public key (1184 bytes)."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != _MLKEM_PK_LEN:
            raise ValueError(f"MLKEMPublicKey must be {_MLKEM_PK_LEN} bytes, got {len(self.raw)}")


@dataclass(frozen=True, slots=True)
class MLKEMKeyPair:
    secret_key: MLKEMSecretKey
    public_key: MLKEMPublicKey


@dataclass(frozen=True, slots=True)
class KEMEncapResult:
    """Result of kem_encapsulate. Send ciphertext to peer; keep shared_secret local."""

    ciphertext: bytes  # 1088 bytes — transmitted to peer
    shared_secret: bytes  # 32 bytes — NEVER transmitted

    def __post_init__(self) -> None:
        if len(self.ciphertext) != _MLKEM_CT_LEN:
            raise ValueError(f"KEMEncapResult.ciphertext must be {_MLKEM_CT_LEN} bytes")
        if len(self.shared_secret) != 32:
            raise ValueError("KEMEncapResult.shared_secret must be 32 bytes")

    def __repr__(self) -> str:
        return f"KEMEncapResult(ciphertext=<{len(self.ciphertext)}B>, shared_secret=<REDACTED>)"


@dataclass(frozen=True, slots=True)
class SessionKey:
    """Derived symmetric session key (32 bytes). Never transmitted."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise ValueError(f"SessionKey must be 32 bytes, got {len(self.raw)}")

    def __repr__(self) -> str:
        return "SessionKey(<REDACTED>)"

    def __str__(self) -> str:
        return "SessionKey(<REDACTED>)"


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def generate_ed25519_keypair() -> Ed25519KeyPair:
    """Generate a new Ed25519 signing keypair."""
    priv = Ed25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return Ed25519KeyPair(
        signing_key=Ed25519SigningKey(raw=priv_raw),
        verify_key=Ed25519VerifyKey(raw=pub_raw),
    )


def generate_x25519_keypair() -> X25519KeyPair:
    """Generate a new X25519 key exchange keypair."""
    priv = X25519PrivateKey.generate()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return X25519KeyPair(
        private_key=X25519PrivKey(raw=priv_raw),
        public_key=X25519PubKey(raw=pub_raw),
    )


def generate_mlkem_keypair() -> MLKEMKeyPair:
    """
    Generate a new ML-KEM-768 keypair for post-quantum key encapsulation.
    Raises RuntimeError if liboqs is not available.
    """
    if not _oqs_available:
        raise RuntimeError(
            "liboqs not available — install the native liboqs library and liboqs-python. "
            "See requirements.txt for build instructions."
        )
    kem = _oqs.KeyEncapsulation(_MLKEM_ALG)
    pk_raw = kem.generate_keypair()
    sk_raw = kem.export_secret_key()
    return MLKEMKeyPair(
        secret_key=MLKEMSecretKey(raw=bytes(sk_raw)),
        public_key=MLKEMPublicKey(raw=bytes(pk_raw)),
    )


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def derive_session_key(
    x25519_shared_secret: bytes,
    mlkem_shared_secret: bytes,
    info: bytes = b"mesim-hybrid-kem-v1\x00session",
    salt: bytes | None = None,
) -> SessionKey:
    """
    Derive a 32-byte session key from X25519 and ML-KEM-768 shared secrets via HKDF-SHA256.

    IKM = x25519_ss || mlkem_ss  (64 bytes total)
    Domain separation is via the `info` parameter.
    """
    ikm = x25519_shared_secret + mlkem_shared_secret
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=info,
    )
    derived = hkdf.derive(ikm)
    return SessionKey(raw=derived)


# ---------------------------------------------------------------------------
# Hybrid KEM
# ---------------------------------------------------------------------------


def kem_encapsulate(
    peer_x25519_pub: X25519PubKey,
    peer_mlkem_pub: MLKEMPublicKey,
    our_x25519_priv: X25519PrivKey,
) -> tuple[KEMEncapResult, SessionKey]:
    """
    Hybrid KEM encapsulation (initiator/sender side).

    Steps:
      1. X25519 ECDH: our_x25519_priv × peer_x25519_pub → x25519_ss
      2. ML-KEM-768 encapsulate(peer_mlkem_pub) → (mlkem_ct, mlkem_ss)
      3. SessionKey = HKDF(x25519_ss || mlkem_ss)

    Returns (KEMEncapResult, SessionKey).
    KEMEncapResult.ciphertext is sent to peer in the handshake.
    SessionKey is used for ChaCha20-Poly1305 and never transmitted.
    """
    if not _oqs_available:
        raise RuntimeError("liboqs not available — ML-KEM-768 required for key exchange")

    # X25519 ECDH
    our_priv = X25519PrivateKey.from_private_bytes(our_x25519_priv.raw)
    peer_pub = _X25519PublicKeyRaw.from_public_bytes(peer_x25519_pub.raw)
    x25519_ss = our_priv.exchange(peer_pub)

    # ML-KEM-768 encapsulation
    kem = _oqs.KeyEncapsulation(_MLKEM_ALG)
    mlkem_ct, mlkem_ss = kem.encap_secret(peer_mlkem_pub.raw)

    session_key = derive_session_key(
        x25519_shared_secret=x25519_ss,
        mlkem_shared_secret=bytes(mlkem_ss),
    )
    encap_result = KEMEncapResult(
        ciphertext=bytes(mlkem_ct),
        shared_secret=bytes(mlkem_ss),
    )
    return encap_result, session_key


def kem_decapsulate(
    kem_ciphertext: bytes,
    initiator_x25519_pub: X25519PubKey,
    our_x25519_priv: X25519PrivKey,
    our_mlkem_secret: MLKEMSecretKey,
) -> SessionKey:
    """
    Hybrid KEM decapsulation (responder side).

    The oqs.KeyEncapsulation object is reconstructed from raw secret key bytes
    rather than stored, to avoid fragile serialization of C objects.
    """
    if not _oqs_available:
        raise RuntimeError("liboqs not available — ML-KEM-768 required for key exchange")

    # X25519 ECDH (symmetric)
    our_priv = X25519PrivateKey.from_private_bytes(our_x25519_priv.raw)
    init_pub = _X25519PublicKeyRaw.from_public_bytes(initiator_x25519_pub.raw)
    x25519_ss = our_priv.exchange(init_pub)

    # ML-KEM-768 decapsulation — reconstruct kem object from secret key bytes
    kem = _oqs.KeyEncapsulation(_MLKEM_ALG, secret_key=our_mlkem_secret.raw)
    mlkem_ss = kem.decap_secret(kem_ciphertext)

    return derive_session_key(
        x25519_shared_secret=x25519_ss,
        mlkem_shared_secret=bytes(mlkem_ss),
    )


# ---------------------------------------------------------------------------
# Symmetric encryption (ChaCha20-Poly1305)
# ---------------------------------------------------------------------------


def encrypt_message(
    plaintext: bytes,
    session_key: SessionKey,
    aad: bytes | None = None,
) -> tuple[bytes, bytes]:
    """
    Encrypt plaintext with ChaCha20-Poly1305.

    Returns (nonce: 12B, ciphertext_with_tag: len(plaintext)+16 B).
    A fresh random 12-byte nonce is generated per call.

    The `aad` parameter binds additional authenticated data (e.g., message header)
    to the ciphertext tag — it must be provided identically on decryption.
    """
    nonce = os.urandom(12)
    chacha = ChaCha20Poly1305(session_key.raw)
    ct = chacha.encrypt(nonce, plaintext, aad)
    return nonce, ct


def decrypt_message(
    nonce: bytes,
    ciphertext: bytes,
    session_key: SessionKey,
    aad: bytes | None = None,
) -> bytes:
    """
    Decrypt and authenticate a ChaCha20-Poly1305 ciphertext.

    Raises cryptography.exceptions.InvalidTag on authentication failure
    (wrong key, tampered ciphertext, wrong nonce, or wrong aad).
    """
    chacha = ChaCha20Poly1305(session_key.raw)
    return chacha.decrypt(nonce, ciphertext, aad)


# ---------------------------------------------------------------------------
# Signing (Ed25519)
# ---------------------------------------------------------------------------


def sign_message(message: bytes, signing_key: Ed25519SigningKey) -> bytes:
    """Sign a message with Ed25519. Returns 64-byte signature."""
    priv = Ed25519PrivateKey.from_private_bytes(signing_key.raw)
    return priv.sign(message)


def verify_signature(
    message: bytes,
    signature: bytes,
    verify_key: Ed25519VerifyKey,
) -> None:
    """
    Verify an Ed25519 signature.

    Raises cryptography.exceptions.InvalidSignature on failure.
    Deliberately returns None — never bool — to prevent accidental
    `if verify_signature(...)` misuse that could pass on exception swallow.
    """
    pub = Ed25519PublicKey.from_public_bytes(verify_key.raw)
    pub.verify(signature, message)


# ---------------------------------------------------------------------------
# NonceCounter — deterministic monotonic nonces (optional, thread-safe)
# ---------------------------------------------------------------------------


class NonceCounter:
    """
    Thread-safe monotonic nonce generator for ChaCha20-Poly1305.

    Format: [4-byte random prefix][8-byte big-endian counter]
    The random prefix scopes the counter to this process instance,
    preventing collision if two processes share a session key.

    Raises OverflowError after 2^64 - 1 calls (never reached in practice).
    """

    def __init__(self) -> None:
        self._prefix = os.urandom(4)
        self._counter = 0
        self._lock = threading.Lock()

    def next_nonce(self) -> bytes:
        with self._lock:
            if self._counter >= (2**64 - 1):
                raise OverflowError("NonceCounter exhausted — rotate session key")
            n = self._counter
            self._counter += 1
        return self._prefix + struct.pack(">Q", n)
