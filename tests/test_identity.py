"""
tests/test_identity.py — Unit tests for core/identity.py

Run: pytest tests/test_identity.py -v
"""

import base64
import json
import uuid

import pytest
from cryptography.exceptions import InvalidSignature, InvalidTag

from core.crypto import _MLKEM_PK_LEN, _MLKEM_SK_LEN, _oqs_available
from core.identity import (
    DeviceIdentity,
    PublicBundle,
    Rank,
    create_identity,
    get_public_bundle,
    load_identity,
    save_identity,
    verify_public_bundle,
    _canonical_bundle_bytes,
)

requires_liboqs = pytest.mark.skipif(not _oqs_available, reason="liboqs not available")


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


def test_rank_ordering():
    assert Rank.COMMAND < Rank.OFFICER < Rank.NCO < Rank.SQUAD


def test_rank_values():
    assert int(Rank.COMMAND) == 1
    assert int(Rank.OFFICER) == 2
    assert int(Rank.NCO) == 3
    assert int(Rank.SQUAD) == 4


def test_rank_roundtrip():
    assert Rank(3) == Rank.NCO


# ---------------------------------------------------------------------------
# create_identity
# ---------------------------------------------------------------------------


@requires_liboqs
def test_create_identity_basic_fields():
    ident = create_identity("ALPHA-1", Rank.NCO)
    assert ident.callsign == "ALPHA-1"
    assert ident.rank == Rank.NCO
    # Valid UUID4
    parsed = uuid.UUID(ident.device_id, version=4)
    assert str(parsed) == ident.device_id


@requires_liboqs
def test_create_identity_auto_uppercase():
    ident = create_identity("bravo-2", Rank.SQUAD)
    assert ident.callsign == "BRAVO-2"


@requires_liboqs
def test_create_identity_key_lengths():
    ident = create_identity("C1", Rank.COMMAND)
    assert len(ident.signing_keypair.signing_key.raw) == 32
    assert len(ident.signing_keypair.verify_key.raw) == 32
    assert len(ident.encrypt_keypair.private_key.raw) == 32
    assert len(ident.encrypt_keypair.public_key.raw) == 32
    assert len(ident.kem_keypair.secret_key.raw) == _MLKEM_SK_LEN
    assert len(ident.kem_keypair.public_key.raw) == _MLKEM_PK_LEN


@requires_liboqs
def test_create_identity_unique_device_ids():
    id1 = create_identity("NODE-A", Rank.SQUAD)
    id2 = create_identity("NODE-B", Rank.SQUAD)
    assert id1.device_id != id2.device_id


@requires_liboqs
def test_create_identity_unique_keypairs():
    id1 = create_identity("A1", Rank.NCO)
    id2 = create_identity("A2", Rank.NCO)
    assert id1.signing_keypair.signing_key.raw != id2.signing_keypair.signing_key.raw
    assert id1.encrypt_keypair.private_key.raw != id2.encrypt_keypair.private_key.raw


def test_create_identity_invalid_callsign_empty():
    with pytest.raises(ValueError, match="Invalid callsign"):
        create_identity("", Rank.NCO)


def test_create_identity_invalid_callsign_too_long():
    with pytest.raises(ValueError, match="Invalid callsign"):
        create_identity("A" * 33, Rank.NCO)


def test_create_identity_invalid_callsign_special_chars():
    with pytest.raises(ValueError, match="Invalid callsign"):
        create_identity("BAD CALLSIGN!", Rank.NCO)


def test_create_identity_invalid_callsign_spaces():
    with pytest.raises(ValueError, match="Invalid callsign"):
        create_identity("BAD CALL", Rank.SQUAD)


def test_create_identity_valid_edge_cases():
    # Single character
    if _oqs_available:
        ident = create_identity("X", Rank.SQUAD)
        assert ident.callsign == "X"
        # Max 32 characters
        ident2 = create_identity("A" * 32, Rank.SQUAD)
        assert len(ident2.callsign) == 32
        # Underscores allowed
        ident3 = create_identity("UNIT_1_ALPHA", Rank.NCO)
        assert ident3.callsign == "UNIT_1_ALPHA"


# ---------------------------------------------------------------------------
# PublicBundle
# ---------------------------------------------------------------------------


@requires_liboqs
def test_get_public_bundle_no_private_material():
    ident = create_identity("ALPHA-1", Rank.OFFICER)
    bundle = get_public_bundle(ident)

    assert isinstance(bundle, PublicBundle)
    assert not hasattr(bundle, "signing_keypair")
    assert not hasattr(bundle, "encrypt_keypair")
    assert not hasattr(bundle, "kem_keypair")


@requires_liboqs
def test_get_public_bundle_fields_match_identity():
    ident = create_identity("BRAVO-3", Rank.COMMAND)
    bundle = get_public_bundle(ident)

    assert bundle.device_id == ident.device_id
    assert bundle.callsign == ident.callsign
    assert bundle.rank == ident.rank
    assert bundle.verify_key.raw == ident.signing_keypair.verify_key.raw
    assert bundle.encrypt_pub.raw == ident.encrypt_keypair.public_key.raw
    assert bundle.kem_pub.raw == ident.kem_keypair.public_key.raw


@requires_liboqs
def test_get_public_bundle_signature_length():
    ident = create_identity("C1", Rank.NCO)
    bundle = get_public_bundle(ident)
    assert len(bundle.bundle_sig) == 64


@requires_liboqs
def test_verify_public_bundle_valid():
    ident = create_identity("DELTA-4", Rank.SQUAD)
    bundle = get_public_bundle(ident)
    verify_public_bundle(bundle)  # must not raise


@requires_liboqs
def test_verify_public_bundle_tampered_callsign():
    ident = create_identity("ECHO-5", Rank.NCO)
    bundle = get_public_bundle(ident)

    # Reconstruct bundle with tampered callsign
    tampered = PublicBundle(
        device_id=bundle.device_id,
        callsign="EVIL-IMPERSONATOR",
        rank=bundle.rank,
        verify_key=bundle.verify_key,
        encrypt_pub=bundle.encrypt_pub,
        kem_pub=bundle.kem_pub,
        bundle_sig=bundle.bundle_sig,
    )
    with pytest.raises(InvalidSignature):
        verify_public_bundle(tampered)


@requires_liboqs
def test_verify_public_bundle_tampered_rank():
    ident = create_identity("FOXTROT-6", Rank.SQUAD)
    bundle = get_public_bundle(ident)

    tampered = PublicBundle(
        device_id=bundle.device_id,
        callsign=bundle.callsign,
        rank=Rank.COMMAND,  # promoted by tampering
        verify_key=bundle.verify_key,
        encrypt_pub=bundle.encrypt_pub,
        kem_pub=bundle.kem_pub,
        bundle_sig=bundle.bundle_sig,
    )
    with pytest.raises(InvalidSignature):
        verify_public_bundle(tampered)


@requires_liboqs
def test_verify_public_bundle_tampered_kem_pub():
    """Verifies that swapping the KEM public key is detected."""
    ident1 = create_identity("GOLF-7", Rank.NCO)
    ident2 = create_identity("HOTEL-8", Rank.NCO)
    bundle = get_public_bundle(ident1)

    tampered = PublicBundle(
        device_id=bundle.device_id,
        callsign=bundle.callsign,
        rank=bundle.rank,
        verify_key=bundle.verify_key,
        encrypt_pub=bundle.encrypt_pub,
        kem_pub=ident2.kem_keypair.public_key,  # KEM oracle attack attempt
        bundle_sig=bundle.bundle_sig,
    )
    with pytest.raises(InvalidSignature):
        verify_public_bundle(tampered)


# ---------------------------------------------------------------------------
# canonical_bundle_bytes
# ---------------------------------------------------------------------------


@requires_liboqs
def test_canonical_bundle_bytes_length():
    ident = create_identity("I1", Rank.NCO)
    canon = _canonical_bundle_bytes(
        device_id=ident.device_id,
        callsign=ident.callsign,
        rank=ident.rank,
        verify_key=ident.signing_keypair.verify_key,
        encrypt_pub=ident.encrypt_keypair.public_key,
        kem_pub=ident.kem_keypair.public_key,
    )
    # 36 + 32 + 1 + 32 + 32 + 1184 = 1317
    assert len(canon) == 1317


@requires_liboqs
def test_canonical_bundle_bytes_deterministic():
    ident = create_identity("J2", Rank.SQUAD)
    b1 = _canonical_bundle_bytes(
        ident.device_id, ident.callsign, ident.rank,
        ident.signing_keypair.verify_key,
        ident.encrypt_keypair.public_key,
        ident.kem_keypair.public_key,
    )
    b2 = _canonical_bundle_bytes(
        ident.device_id, ident.callsign, ident.rank,
        ident.signing_keypair.verify_key,
        ident.encrypt_keypair.public_key,
        ident.kem_keypair.public_key,
    )
    assert b1 == b2


# ---------------------------------------------------------------------------
# save_identity / load_identity
# ---------------------------------------------------------------------------


@requires_liboqs
def test_save_load_roundtrip(tmp_path):
    ident = create_identity("KILO-9", Rank.OFFICER)
    path = tmp_path / "kilo.json"
    save_identity(ident, path, "correct-passphrase")
    loaded = load_identity(path, "correct-passphrase")

    assert loaded.device_id == ident.device_id
    assert loaded.callsign == ident.callsign
    assert loaded.rank == ident.rank
    assert loaded.signing_keypair.signing_key.raw == ident.signing_keypair.signing_key.raw
    assert loaded.signing_keypair.verify_key.raw == ident.signing_keypair.verify_key.raw
    assert loaded.encrypt_keypair.private_key.raw == ident.encrypt_keypair.private_key.raw
    assert loaded.encrypt_keypair.public_key.raw == ident.encrypt_keypair.public_key.raw
    assert loaded.kem_keypair.secret_key.raw == ident.kem_keypair.secret_key.raw
    assert loaded.kem_keypair.public_key.raw == ident.kem_keypair.public_key.raw


@requires_liboqs
def test_save_load_passphrase_as_bytes(tmp_path):
    ident = create_identity("LIMA-10", Rank.SQUAD)
    path = tmp_path / "lima.json"
    save_identity(ident, path, b"bytes-passphrase")
    loaded = load_identity(path, b"bytes-passphrase")
    assert loaded.device_id == ident.device_id


@requires_liboqs
def test_load_wrong_passphrase_raises(tmp_path):
    ident = create_identity("MIKE-11", Rank.NCO)
    path = tmp_path / "mike.json"
    save_identity(ident, path, "correct")
    with pytest.raises(InvalidTag):
        load_identity(path, "wrong")


@requires_liboqs
def test_load_corrupted_encrypted_block_raises(tmp_path):
    ident = create_identity("NOVEMBER-12", Rank.SQUAD)
    path = tmp_path / "november.json"
    save_identity(ident, path, "passphrase")

    doc = json.loads(path.read_text())
    # Corrupt the last byte of the encrypted private key block
    enc = base64.b64decode(doc["encrypted_private_keys"])
    corrupted = enc[:-1] + bytes([enc[-1] ^ 0xFF])
    doc["encrypted_private_keys"] = base64.b64encode(corrupted).decode()
    path.write_text(json.dumps(doc))

    with pytest.raises(InvalidTag):
        load_identity(path, "passphrase")


@requires_liboqs
def test_load_version_mismatch_raises(tmp_path):
    ident = create_identity("OSCAR-13", Rank.SQUAD)
    path = tmp_path / "oscar.json"
    save_identity(ident, path, "pw")

    doc = json.loads(path.read_text())
    doc["version"] = 99
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="unsupported identity file version"):
        load_identity(path, "pw")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_identity(tmp_path / "nonexistent.json", "pw")


@requires_liboqs
def test_identity_file_contains_no_plaintext_private_keys(tmp_path):
    ident = create_identity("PAPA-14", Rank.NCO)
    path = tmp_path / "papa.json"
    save_identity(ident, path, "secure-passphrase")

    file_content = path.read_text()

    # Raw private key material must not appear in the JSON file as hex or base64
    sk_hex = ident.signing_keypair.signing_key.raw.hex()
    xk_hex = ident.encrypt_keypair.private_key.raw.hex()
    assert sk_hex not in file_content
    assert xk_hex not in file_content

    # Also check first 16 bytes of ML-KEM secret key
    mlkem_hex_prefix = ident.kem_keypair.secret_key.raw[:16].hex()
    assert mlkem_hex_prefix not in file_content


@requires_liboqs
def test_saved_identity_is_valid_json(tmp_path):
    ident = create_identity("QUEBEC-15", Rank.SQUAD)
    path = tmp_path / "quebec.json"
    save_identity(ident, path, "pw")
    doc = json.loads(path.read_text())
    assert doc["version"] == 1
    assert "encrypted_private_keys" in doc
    assert "public_keys" in doc
    assert "kdf" in doc


@requires_liboqs
def test_loaded_identity_bundle_verifies(tmp_path):
    """End-to-end: save, load, get_public_bundle, verify_public_bundle."""
    ident = create_identity("ROMEO-16", Rank.OFFICER)
    path = tmp_path / "romeo.json"
    save_identity(ident, path, "pw")
    loaded = load_identity(path, "pw")

    bundle = get_public_bundle(loaded)
    verify_public_bundle(bundle)  # must not raise


@requires_liboqs
def test_save_overwrites_existing_file(tmp_path):
    ident = create_identity("SIERRA-17", Rank.SQUAD)
    path = tmp_path / "sierra.json"
    save_identity(ident, path, "pw1")
    save_identity(ident, path, "pw2")  # overwrite
    loaded = load_identity(path, "pw2")
    assert loaded.device_id == ident.device_id
    with pytest.raises(InvalidTag):
        load_identity(path, "pw1")


# ---------------------------------------------------------------------------
# DeviceIdentity type
# ---------------------------------------------------------------------------


@requires_liboqs
def test_device_identity_is_mutable():
    """DeviceIdentity is not frozen — rank can be updated by COMMAND node."""
    ident = create_identity("TANGO-18", Rank.SQUAD)
    ident.rank = Rank.NCO
    assert ident.rank == Rank.NCO


# ---------------------------------------------------------------------------
# Duress PIN
# ---------------------------------------------------------------------------


@requires_liboqs
def test_real_passphrase_still_loads_real_identity_with_duress_configured(tmp_path):
    """Real passphrase returns real identity even when duress PIN is set."""
    ident = create_identity("UNIFORM-19", Rank.COMMAND)
    path = tmp_path / "uniform.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    loaded = load_identity(path, "real_pass")
    assert loaded.device_id == ident.device_id
    assert loaded.callsign == ident.callsign


@requires_liboqs
def test_duress_passphrase_returns_decoy_identity(tmp_path):
    """Duress passphrase returns a decoy with same callsign/rank but different device_id."""
    ident = create_identity("VICTOR-20", Rank.NCO)
    path = tmp_path / "victor.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    decoy = load_identity(path, "duress_pass")
    assert decoy.callsign == ident.callsign
    assert decoy.rank == ident.rank
    assert decoy.device_id != ident.device_id


@requires_liboqs
def test_duress_wipes_real_keys_from_disk(tmp_path):
    """After duress activation, real passphrase no longer decrypts the file."""
    ident = create_identity("WHISKEY-21", Rank.OFFICER)
    path = tmp_path / "whiskey.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    load_identity(path, "duress_pass")  # activates duress → wipe
    with pytest.raises(InvalidTag):
        load_identity(path, "real_pass")


@requires_liboqs
def test_duress_file_has_no_duress_section_after_wipe(tmp_path):
    """The identity file after duress activation looks like a normal file (no duress fields)."""
    ident = create_identity("XRAY-22", Rank.SQUAD)
    path = tmp_path / "xray.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    load_identity(path, "duress_pass")  # activate duress
    doc = json.loads(path.read_text())
    assert "duress_encrypted_keys" not in doc
    assert "duress_salt" not in doc


@requires_liboqs
def test_decoy_loadable_after_wipe_with_duress_passphrase(tmp_path):
    """After wipe, duress passphrase loads the same decoy identity consistently."""
    ident = create_identity("YANKEE-23", Rank.SQUAD)
    path = tmp_path / "yankee.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    decoy1 = load_identity(path, "duress_pass")   # first load — activates wipe
    decoy2 = load_identity(path, "duress_pass")   # second load — from wiped file
    assert decoy1.device_id == decoy2.device_id


@requires_liboqs
def test_wrong_passphrase_raises_invalid_tag_with_duress_configured(tmp_path):
    """Wrong passphrase raises InvalidTag even when duress is configured."""
    ident = create_identity("ZULU-24", Rank.NCO)
    path = tmp_path / "zulu.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    with pytest.raises(InvalidTag):
        load_identity(path, "totally_wrong")


@requires_liboqs
def test_save_without_duress_has_no_duress_fields(tmp_path):
    """Identity file saved without duress_passphrase contains no duress fields."""
    ident = create_identity("ALPHA-99", Rank.SQUAD)
    path = tmp_path / "a99.json"
    save_identity(ident, path, "pass")
    doc = json.loads(path.read_text())
    assert "duress_encrypted_keys" not in doc
    assert "duress_salt" not in doc
    assert "duress_device_id" not in doc


@requires_liboqs
def test_decoy_identity_has_valid_keypairs(tmp_path):
    """Decoy identity returned by duress path has working keypairs (verifiable bundle)."""
    ident = create_identity("BRAVO-99", Rank.COMMAND)
    path = tmp_path / "b99.json"
    save_identity(ident, path, "real_pass", duress_passphrase="duress_pass")
    decoy = load_identity(path, "duress_pass")
    bundle = get_public_bundle(decoy)
    verify_public_bundle(bundle)  # must not raise
