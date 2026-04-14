"""
tests/test_crypto.py — Unit tests for core/crypto.py

Run: pytest tests/test_crypto.py -v
ML-KEM-768 tests are skipped automatically if liboqs is not available.
"""

import pytest
from cryptography.exceptions import InvalidSignature, InvalidTag

from core.crypto import (
    Ed25519KeyPair,
    Ed25519SigningKey,
    Ed25519VerifyKey,
    KEMEncapResult,
    MLKEMKeyPair,
    MLKEMPublicKey,
    MLKEMSecretKey,
    NonceCounter,
    SessionKey,
    X25519KeyPair,
    X25519PrivKey,
    X25519PubKey,
    _MLKEM_CT_LEN,
    _MLKEM_PK_LEN,
    _MLKEM_SK_LEN,
    _oqs_available,
    decrypt_message,
    derive_session_key,
    encrypt_message,
    generate_ed25519_keypair,
    generate_mlkem_keypair,
    generate_x25519_keypair,
    kem_decapsulate,
    kem_encapsulate,
    sign_message,
    verify_signature,
)

requires_liboqs = pytest.mark.skipif(not _oqs_available, reason="liboqs not available")


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def test_generate_ed25519_keypair_types_and_lengths():
    kp = generate_ed25519_keypair()
    assert isinstance(kp, Ed25519KeyPair)
    assert len(kp.signing_key.raw) == 32
    assert len(kp.verify_key.raw) == 32


def test_generate_x25519_keypair_types_and_lengths():
    kp = generate_x25519_keypair()
    assert isinstance(kp, X25519KeyPair)
    assert len(kp.private_key.raw) == 32
    assert len(kp.public_key.raw) == 32


def test_generate_ed25519_keypairs_are_unique():
    kp1 = generate_ed25519_keypair()
    kp2 = generate_ed25519_keypair()
    assert kp1.signing_key.raw != kp2.signing_key.raw


def test_generate_x25519_keypairs_are_unique():
    kp1 = generate_x25519_keypair()
    kp2 = generate_x25519_keypair()
    assert kp1.private_key.raw != kp2.private_key.raw


@requires_liboqs
def test_generate_mlkem_keypair_types_and_lengths():
    kp = generate_mlkem_keypair()
    assert isinstance(kp, MLKEMKeyPair)
    assert len(kp.public_key.raw) == _MLKEM_PK_LEN
    assert len(kp.secret_key.raw) == _MLKEM_SK_LEN


@requires_liboqs
def test_generate_mlkem_keypairs_are_unique():
    kp1 = generate_mlkem_keypair()
    kp2 = generate_mlkem_keypair()
    assert kp1.public_key.raw != kp2.public_key.raw


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def test_ed25519_sign_verify_roundtrip():
    kp = generate_ed25519_keypair()
    msg = b"ALPHA-1 to BRAVO-2: grid 442, move now"
    sig = sign_message(msg, kp.signing_key)
    assert len(sig) == 64
    verify_signature(msg, sig, kp.verify_key)  # must not raise


def test_ed25519_verify_tampered_message_raises():
    kp = generate_ed25519_keypair()
    sig = sign_message(b"original", kp.signing_key)
    with pytest.raises(InvalidSignature):
        verify_signature(b"tampered", sig, kp.verify_key)


def test_ed25519_verify_wrong_key_raises():
    kp1 = generate_ed25519_keypair()
    kp2 = generate_ed25519_keypair()
    sig = sign_message(b"secret", kp1.signing_key)
    with pytest.raises(InvalidSignature):
        verify_signature(b"secret", sig, kp2.verify_key)


def test_ed25519_verify_truncated_signature_raises():
    kp = generate_ed25519_keypair()
    sig = sign_message(b"msg", kp.signing_key)
    with pytest.raises(Exception):
        verify_signature(b"msg", sig[:32], kp.verify_key)


def test_ed25519_verify_returns_none():
    kp = generate_ed25519_keypair()
    sig = sign_message(b"test", kp.signing_key)
    result = verify_signature(b"test", sig, kp.verify_key)
    assert result is None  # must not return bool True


# ---------------------------------------------------------------------------
# Symmetric encryption
# ---------------------------------------------------------------------------


def _make_session_key() -> SessionKey:
    import os
    return SessionKey(raw=os.urandom(32))


def test_encrypt_decrypt_roundtrip():
    sk = _make_session_key()
    plaintext = b"classified: rally at dawn"
    nonce, ct = encrypt_message(plaintext, sk)
    assert len(nonce) == 12
    assert len(ct) == len(plaintext) + 16  # +16 for Poly1305 tag
    recovered = decrypt_message(nonce, ct, sk)
    assert recovered == plaintext


def test_encrypt_decrypt_with_aad():
    sk = _make_session_key()
    plaintext = b"move to objective"
    aad = b"seq:00001"
    nonce, ct = encrypt_message(plaintext, sk, aad=aad)
    recovered = decrypt_message(nonce, ct, sk, aad=aad)
    assert recovered == plaintext


def test_decrypt_wrong_aad_raises():
    sk = _make_session_key()
    nonce, ct = encrypt_message(b"data", sk, aad=b"correct")
    with pytest.raises(InvalidTag):
        decrypt_message(nonce, ct, sk, aad=b"wrong")


def test_decrypt_no_aad_when_encrypted_with_aad_raises():
    sk = _make_session_key()
    nonce, ct = encrypt_message(b"data", sk, aad=b"header")
    with pytest.raises(InvalidTag):
        decrypt_message(nonce, ct, sk, aad=None)


def test_decrypt_tampered_ciphertext_raises():
    sk = _make_session_key()
    nonce, ct = encrypt_message(b"sensitive", sk)
    tampered = ct[:-1] + bytes([ct[-1] ^ 0xFF])
    with pytest.raises(InvalidTag):
        decrypt_message(nonce, tampered, sk)


def test_decrypt_tampered_nonce_raises():
    sk = _make_session_key()
    nonce, ct = encrypt_message(b"sensitive", sk)
    bad_nonce = bytes([nonce[0] ^ 0x01]) + nonce[1:]
    with pytest.raises(InvalidTag):
        decrypt_message(bad_nonce, ct, sk)


def test_decrypt_wrong_key_raises():
    sk1 = _make_session_key()
    sk2 = _make_session_key()
    nonce, ct = encrypt_message(b"data", sk1)
    with pytest.raises(InvalidTag):
        decrypt_message(nonce, ct, sk2)


def test_unique_nonces_per_call():
    sk = _make_session_key()
    nonces = {encrypt_message(b"x", sk)[0] for _ in range(1000)}
    assert len(nonces) == 1000


def test_encrypt_empty_plaintext():
    sk = _make_session_key()
    nonce, ct = encrypt_message(b"", sk)
    assert decrypt_message(nonce, ct, sk) == b""


# ---------------------------------------------------------------------------
# Hybrid KEM
# ---------------------------------------------------------------------------


@requires_liboqs
def test_kem_encap_decap_equal_session_keys():
    """Full hybrid KEM roundtrip: Alice encapsulates to Bob, Bob decapsulates."""
    alice_x = generate_x25519_keypair()
    bob_x = generate_x25519_keypair()
    bob_kem = generate_mlkem_keypair()

    # Alice encapsulates to Bob
    encap_result, alice_session = kem_encapsulate(
        peer_x25519_pub=bob_x.public_key,
        peer_mlkem_pub=bob_kem.public_key,
        our_x25519_priv=alice_x.private_key,
    )

    # Bob decapsulates
    bob_session = kem_decapsulate(
        kem_ciphertext=encap_result.ciphertext,
        initiator_x25519_pub=alice_x.public_key,
        our_x25519_priv=bob_x.private_key,
        our_mlkem_secret=bob_kem.secret_key,
    )

    assert alice_session.raw == bob_session.raw


@requires_liboqs
def test_kem_ciphertext_length():
    alice_x = generate_x25519_keypair()
    bob_x = generate_x25519_keypair()
    bob_kem = generate_mlkem_keypair()

    encap_result, _ = kem_encapsulate(
        peer_x25519_pub=bob_x.public_key,
        peer_mlkem_pub=bob_kem.public_key,
        our_x25519_priv=alice_x.private_key,
    )
    assert len(encap_result.ciphertext) == _MLKEM_CT_LEN


@requires_liboqs
def test_kem_tampered_ciphertext_produces_different_key():
    """ML-KEM-768 is IND-CCA2: tampered ciphertext decapsulates to different key."""
    alice_x = generate_x25519_keypair()
    bob_x = generate_x25519_keypair()
    bob_kem = generate_mlkem_keypair()

    encap_result, alice_session = kem_encapsulate(
        peer_x25519_pub=bob_x.public_key,
        peer_mlkem_pub=bob_kem.public_key,
        our_x25519_priv=alice_x.private_key,
    )

    tampered_ct = bytes([encap_result.ciphertext[0] ^ 0xFF]) + encap_result.ciphertext[1:]
    bob_session_bad = kem_decapsulate(
        kem_ciphertext=tampered_ct,
        initiator_x25519_pub=alice_x.public_key,
        our_x25519_priv=bob_x.private_key,
        our_mlkem_secret=bob_kem.secret_key,
    )

    assert alice_session.raw != bob_session_bad.raw


@requires_liboqs
def test_full_encrypted_message_roundtrip_with_kem():
    """KEM → session key → sign-then-encrypt → decrypt → verify sig."""
    alice_x = generate_x25519_keypair()
    alice_ed = generate_ed25519_keypair()
    bob_x = generate_x25519_keypair()
    bob_kem = generate_mlkem_keypair()

    _, session = kem_encapsulate(
        peer_x25519_pub=bob_x.public_key,
        peer_mlkem_pub=bob_kem.public_key,
        our_x25519_priv=alice_x.private_key,
    )

    plaintext = b"fire mission: grid 4427"
    sig = sign_message(plaintext, alice_ed.signing_key)
    nonce, ct = encrypt_message(sig + plaintext, session)
    decrypted = decrypt_message(nonce, ct, session)

    recovered_sig = decrypted[:64]
    recovered_plaintext = decrypted[64:]
    assert recovered_plaintext == plaintext
    verify_signature(recovered_plaintext, recovered_sig, alice_ed.verify_key)


# ---------------------------------------------------------------------------
# Key type safety
# ---------------------------------------------------------------------------


def test_private_key_repr_redacted():
    assert "<REDACTED>" in repr(Ed25519SigningKey(raw=b"\x00" * 32))
    assert "<REDACTED>" in repr(X25519PrivKey(raw=b"\x00" * 32))
    assert "<REDACTED>" in repr(SessionKey(raw=b"\x00" * 32))


def test_private_key_str_redacted():
    assert "<REDACTED>" in str(Ed25519SigningKey(raw=b"\x00" * 32))


@requires_liboqs
def test_mlkem_secret_key_repr_redacted():
    assert "<REDACTED>" in repr(MLKEMSecretKey(raw=b"\x00" * _MLKEM_SK_LEN))


def test_public_keys_are_not_redacted():
    vk = Ed25519VerifyKey(raw=b"\xab" * 32)
    assert "REDACTED" not in repr(vk)


def test_ed25519_signing_key_length_validation():
    with pytest.raises(ValueError):
        Ed25519SigningKey(raw=b"\x00" * 31)


def test_x25519_priv_key_length_validation():
    with pytest.raises(ValueError):
        X25519PrivKey(raw=b"\x00" * 33)


def test_x25519_pub_key_length_validation():
    with pytest.raises(ValueError):
        X25519PubKey(raw=b"\x00" * 31)


@requires_liboqs
def test_mlkem_public_key_length_validation():
    with pytest.raises(ValueError):
        MLKEMPublicKey(raw=b"\x00" * (_MLKEM_PK_LEN - 1))


@requires_liboqs
def test_mlkem_secret_key_length_validation():
    with pytest.raises(ValueError):
        MLKEMSecretKey(raw=b"\x00" * (_MLKEM_SK_LEN + 1))


def test_session_key_length_validation():
    with pytest.raises(ValueError):
        SessionKey(raw=b"\x00" * 16)


# ---------------------------------------------------------------------------
# HKDF domain separation
# ---------------------------------------------------------------------------


def test_derive_session_key_different_info_different_keys():
    import os
    x_ss = os.urandom(32)
    m_ss = os.urandom(32)
    k1 = derive_session_key(x_ss, m_ss, info=b"mesim-hybrid-kem-v1\x00session")
    k2 = derive_session_key(x_ss, m_ss, info=b"mesim-hybrid-kem-v1\x00ratchet")
    assert k1.raw != k2.raw


def test_derive_session_key_is_deterministic():
    import os
    x_ss = os.urandom(32)
    m_ss = os.urandom(32)
    k1 = derive_session_key(x_ss, m_ss)
    k2 = derive_session_key(x_ss, m_ss)
    assert k1.raw == k2.raw


# ---------------------------------------------------------------------------
# NonceCounter
# ---------------------------------------------------------------------------


def test_nonce_counter_produces_12_byte_nonces():
    nc = NonceCounter()
    assert len(nc.next_nonce()) == 12


def test_nonce_counter_monotonic_and_unique():
    nc = NonceCounter()
    nonces = [nc.next_nonce() for _ in range(500)]
    assert len(set(nonces)) == 500


def test_nonce_counter_prefix_consistent():
    nc = NonceCounter()
    n1 = nc.next_nonce()
    n2 = nc.next_nonce()
    assert n1[:4] == n2[:4]  # same prefix within an instance
    assert n1[4:] != n2[4:]  # counter advances


def test_nonce_counter_two_instances_different_prefix():
    nc1 = NonceCounter()
    nc2 = NonceCounter()
    # Overwhelmingly likely to differ (4-byte random prefix)
    assert nc1.next_nonce()[:4] != nc2.next_nonce()[:4]


def test_nonce_counter_thread_safe():
    import threading
    nc = NonceCounter()
    results = []
    lock = threading.Lock()

    def collect():
        for _ in range(200):
            n = nc.next_nonce()
            with lock:
                results.append(n)

    threads = [threading.Thread(target=collect) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(results)) == 1000  # all unique
