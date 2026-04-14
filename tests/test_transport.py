"""
tests/test_transport.py — Unit and integration tests for mesh/transport.py

Groups 1–7: pure unit tests (no network).
Groups 8–11: real asyncio UDP on 127.0.0.1 loopback sockets.

Run: pytest tests/test_transport.py -v
"""

import asyncio
import os
import struct
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from core.crypto import (
    InvalidTag,
    NonceCounter,
    SessionKey,
    _oqs_available,
    decrypt_message,
    encrypt_message,
    sign_message,
)
from core.identity import DeviceIdentity, PublicBundle, Rank, create_identity, get_public_bundle

requires_liboqs = pytest.mark.skipif(not _oqs_available, reason="liboqs not available")


# ---------------------------------------------------------------------------
# Lazy import (module does not exist until we build it)
# ---------------------------------------------------------------------------

def _import():
    from mesh.transport import (
        MeshTransport,
        PacketType,
        SessionInfo,
        HEADER_FMT,
        HEADER_SIZE,
        MAC_SIZE,
        OVERHEAD,
        MTU,
        MAX_PAYLOAD,
        BUNDLE_WIRE_LEN,
        PROTOCOL_VERSION,
        ZERO_MAC,
        KEY_ACK_MAGIC,
        FRAG_HEADER_SIZE,
        build_packet,
        parse_packet,
        pack_bundle,
        unpack_bundle,
        compute_mac,
        verify_mac,
        sender_id_bytes,
        device_id_from_bytes,
    )
    return locals()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def identity_a():
    return create_identity("ALPHA-1", Rank.NCO)


@pytest_asyncio.fixture
async def identity_b():
    return create_identity("BRAVO-2", Rank.SQUAD)


@pytest_asyncio.fixture
async def bundle_a(identity_a):
    return get_public_bundle(identity_a)


@pytest_asyncio.fixture
async def bundle_b(identity_b):
    return get_public_bundle(identity_b)


@pytest_asyncio.fixture
async def session_key():
    return SessionKey(raw=os.urandom(32))


@pytest_asyncio.fixture
async def transport_a(identity_a):
    from mesh.transport import MeshTransport
    t = MeshTransport(identity_a)
    await t.start("127.0.0.1", 0)
    yield t
    await t.stop()


@pytest_asyncio.fixture
async def transport_b(identity_b):
    from mesh.transport import MeshTransport
    t = MeshTransport(identity_b)
    await t.start("127.0.0.1", 0)
    yield t
    await t.stop()


def _addr_of(transport) -> tuple[str, int]:
    """Get the bound address of a MeshTransport."""
    return transport._udp_transport.get_extra_info("sockname")


# ---------------------------------------------------------------------------
# Group 1 — Constants and PacketType
# ---------------------------------------------------------------------------


def test_packet_type_values():
    m = _import()
    PacketType = m["PacketType"]
    assert PacketType.HANDSHAKE == 0x01
    assert PacketType.HANDSHAKE_RESP == 0x02
    assert PacketType.KEY_INIT == 0x03
    assert PacketType.KEY_ACK == 0x04
    assert PacketType.MESSAGE == 0x05
    assert PacketType.ACK == 0x06
    assert PacketType.PING == 0x07
    assert PacketType.PONG == 0x08
    assert PacketType.STORE_FORWARD == 0x09


def test_header_size_is_36():
    m = _import()
    assert struct.calcsize(m["HEADER_FMT"]) == 36
    assert m["HEADER_SIZE"] == 36


def test_max_payload_is_mtu_minus_overhead():
    m = _import()
    assert m["MTU"] - m["OVERHEAD"] == m["MAX_PAYLOAD"]
    assert m["MAX_PAYLOAD"] == 1348


def test_bundle_wire_len_is_1381():
    m = _import()
    # device_id(36) + verify_key(32) + encrypt_pub(32) + kem_pub(1184) + callsign(32) + rank(1) + sig(64)
    assert 36 + 32 + 32 + 1184 + 32 + 1 + 64 == 1381
    assert m["BUNDLE_WIRE_LEN"] == 1381


# ---------------------------------------------------------------------------
# Group 2 — pack_bundle / unpack_bundle
# ---------------------------------------------------------------------------


@requires_liboqs
def test_pack_bundle_length(bundle_a):
    m = _import()
    assert len(m["pack_bundle"](bundle_a)) == m["BUNDLE_WIRE_LEN"]


@requires_liboqs
def test_unpack_bundle_roundtrip(bundle_a):
    m = _import()
    packed = m["pack_bundle"](bundle_a)
    unpacked = m["unpack_bundle"](packed)
    assert unpacked.verify_key.raw == bundle_a.verify_key.raw
    assert unpacked.encrypt_pub.raw == bundle_a.encrypt_pub.raw
    assert unpacked.kem_pub.raw == bundle_a.kem_pub.raw
    assert unpacked.callsign == bundle_a.callsign
    assert unpacked.rank == bundle_a.rank
    assert unpacked.bundle_sig == bundle_a.bundle_sig


@requires_liboqs
def test_unpack_bundle_callsign_strip_padding(bundle_a):
    m = _import()
    packed = m["pack_bundle"](bundle_a)
    unpacked = m["unpack_bundle"](packed)
    assert "\x00" not in unpacked.callsign
    assert unpacked.callsign == bundle_a.callsign


def test_unpack_bundle_wrong_length_raises():
    m = _import()
    with pytest.raises(ValueError):
        m["unpack_bundle"](b"\x00" * 100)


@requires_liboqs
def test_unpacked_bundle_passes_verify_public_bundle(bundle_a):
    from core.identity import verify_public_bundle
    m = _import()
    packed = m["pack_bundle"](bundle_a)
    unpacked = m["unpack_bundle"](packed)
    verify_public_bundle(unpacked)  # must not raise


# ---------------------------------------------------------------------------
# Group 3 — build_packet / parse_packet
# ---------------------------------------------------------------------------


def test_build_packet_total_length(session_key):
    m = _import()
    payload = b"A" * 100
    pkt = m["build_packet"](
        m["PacketType"].MESSAGE,
        b"a" * 32,
        payload,
        session_key,
    )
    assert len(pkt) == m["OVERHEAD"] + len(payload)


def test_build_packet_handshake_has_zero_mac():
    m = _import()
    pkt = m["build_packet"](
        m["PacketType"].HANDSHAKE,
        b"a" * 32,
        b"bundle_data",
        session_key=None,
    )
    assert pkt[-16:] == m["ZERO_MAC"]


def test_build_packet_message_has_nonzero_mac(session_key):
    m = _import()
    pkt = m["build_packet"](
        m["PacketType"].MESSAGE,
        b"a" * 32,
        b"encrypted_payload",
        session_key,
    )
    assert pkt[-16:] != m["ZERO_MAC"]


def test_build_parse_packet_roundtrip(session_key):
    m = _import()
    sid = b"f" * 32
    payload = b"test payload contents"
    pkt = m["build_packet"](m["PacketType"].MESSAGE, sid, payload, session_key)
    version, ptype, parsed_sid, parsed_payload = m["parse_packet"](pkt)
    assert version == m["PROTOCOL_VERSION"]
    assert ptype == m["PacketType"].MESSAGE
    assert parsed_sid == sid
    assert parsed_payload == payload


def test_parse_packet_too_short_raises():
    m = _import()
    with pytest.raises(ValueError):
        m["parse_packet"](b"\x01\x05" + b"a" * 10)


def test_parse_packet_wrong_version_raises():
    m = _import()
    bad = struct.pack("!BB32sH", 99, 0x05, b"a" * 32, 0) + b"\x00" * 16
    with pytest.raises(ValueError):
        m["parse_packet"](bad)


# ---------------------------------------------------------------------------
# Group 4 — compute_mac / verify_mac
# ---------------------------------------------------------------------------


def test_compute_mac_returns_16_bytes(session_key):
    m = _import()
    header = b"H" * 36
    payload = b"P" * 50
    mac = m["compute_mac"](session_key, header, payload)
    assert len(mac) == 16


def test_verify_mac_valid_passes(session_key):
    m = _import()
    pkt = m["build_packet"](m["PacketType"].MESSAGE, b"s" * 32, b"data", session_key)
    m["verify_mac"](session_key, pkt)  # must not raise


def test_verify_mac_tampered_payload_raises(session_key):
    m = _import()
    pkt = m["build_packet"](m["PacketType"].MESSAGE, b"s" * 32, b"data___", session_key)
    # Flip a byte in the payload area
    pkt_list = bytearray(pkt)
    pkt_list[36] ^= 0xFF
    with pytest.raises(InvalidTag):
        m["verify_mac"](session_key, bytes(pkt_list))


# ---------------------------------------------------------------------------
# Group 5 — sender_id_bytes / device_id_from_bytes
# ---------------------------------------------------------------------------


@requires_liboqs
def test_sender_id_bytes_length_and_encoding(identity_a):
    m = _import()
    sid = m["sender_id_bytes"](identity_a.device_id)
    assert len(sid) == 32
    assert b"-" not in sid
    sid.decode("ascii")  # must be valid ASCII


@requires_liboqs
def test_device_id_round_trip(identity_a):
    m = _import()
    sid = m["sender_id_bytes"](identity_a.device_id)
    recovered = m["device_id_from_bytes"](sid)
    assert recovered == identity_a.device_id.replace("-", "")[:32]


# ---------------------------------------------------------------------------
# Group 6 — SessionInfo dataclass
# ---------------------------------------------------------------------------


@requires_liboqs
def test_session_info_fields(identity_b, bundle_b, session_key):
    m = _import()
    now = time.time()
    si = m["SessionInfo"](
        peer_id=identity_b.device_id,
        peer_addr=("127.0.0.1", 9999),
        session_key=session_key,
        nonce_counter=NonceCounter(),
        peer_bundle=bundle_b,
        established_at=now,
        last_active=now,
    )
    assert si.peer_id == identity_b.device_id
    assert si.peer_addr == ("127.0.0.1", 9999)
    assert si.session_key is session_key
    assert si.peer_bundle is bundle_b


@requires_liboqs
def test_session_info_bytes_default_zero(identity_b, bundle_b, session_key):
    m = _import()
    now = time.time()
    si = m["SessionInfo"](
        peer_id=identity_b.device_id,
        peer_addr=("127.0.0.1", 9999),
        session_key=session_key,
        nonce_counter=NonceCounter(),
        peer_bundle=bundle_b,
        established_at=now,
        last_active=now,
    )
    assert si.bytes_sent == 0
    assert si.bytes_recv == 0


# ---------------------------------------------------------------------------
# Group 7 — MeshTransport start / stop
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_transport_start_binds_socket(identity_a):
    from mesh.transport import MeshTransport
    t = MeshTransport(identity_a)
    await t.start("127.0.0.1", 0)
    assert t._udp_transport is not None
    await t.stop()


@requires_liboqs
async def test_transport_stop_cleans_up(identity_a):
    from mesh.transport import MeshTransport
    t = MeshTransport(identity_a)
    await t.start("127.0.0.1", 0)
    await t.stop()
    assert t._udp_transport is None


# ---------------------------------------------------------------------------
# Group 8 — Full handshake integration (real UDP loopback)
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_connect_returns_true_on_successful_handshake(
    transport_a, transport_b, bundle_b
):
    addr_b = _addr_of(transport_b)
    result = await asyncio.wait_for(
        transport_a.connect(addr_b, bundle_b), timeout=5.0
    )
    assert result is True


@requires_liboqs
async def test_handshake_both_sides_have_session(
    transport_a, transport_b, bundle_b
):
    addr_b = _addr_of(transport_b)
    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    # Give responder a moment to finish
    await asyncio.sleep(0.1)
    assert len(transport_a.get_sessions()) == 1
    assert len(transport_b.get_sessions()) == 1


@requires_liboqs
async def test_handshake_session_keys_match(
    transport_a, transport_b, bundle_b
):
    addr_b = _addr_of(transport_b)
    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.1)

    sess_a = list(transport_a.get_sessions().values())[0]
    sess_b = list(transport_b.get_sessions().values())[0]
    assert sess_a.session_key.raw == sess_b.session_key.raw


@requires_liboqs
async def test_connect_invalid_bundle_raises(identity_a, identity_b, bundle_b):
    from mesh.transport import MeshTransport
    from cryptography.exceptions import InvalidSignature

    t = MeshTransport(identity_a)
    await t.start("127.0.0.1", 0)
    try:
        # Tamper the bundle
        from core.identity import PublicBundle
        tampered = PublicBundle(
            device_id=bundle_b.device_id,
            callsign="EVIL",
            rank=bundle_b.rank,
            verify_key=bundle_b.verify_key,
            encrypt_pub=bundle_b.encrypt_pub,
            kem_pub=bundle_b.kem_pub,
            bundle_sig=bundle_b.bundle_sig,
        )
        with pytest.raises(InvalidSignature):
            await t.connect(("127.0.0.1", 9998), tampered)
    finally:
        await t.stop()


# ---------------------------------------------------------------------------
# Group 9 — MESSAGE send/receive (real UDP loopback)
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_send_message_received_by_peer(
    transport_a, transport_b, bundle_b
):
    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    messages = []

    async def on_msg(sender_id, plaintext):
        messages.append(plaintext)
        received.set()

    transport_b.on_message(on_msg)
    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    b_device_id = list(transport_a.get_sessions().keys())[0]
    await transport_a.send(b_device_id, b"alpha to bravo: confirmed")

    await asyncio.wait_for(received.wait(), timeout=5.0)
    assert b"alpha to bravo: confirmed" in messages


@requires_liboqs
async def test_message_callback_receives_correct_sender_id(
    transport_a, transport_b, bundle_b, identity_a
):
    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    sender_ids = []

    async def on_msg(sender_id, plaintext):
        sender_ids.append(sender_id)
        received.set()

    transport_b.on_message(on_msg)
    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    b_device_id = list(transport_a.get_sessions().keys())[0]
    await transport_a.send(b_device_id, b"test")
    await asyncio.wait_for(received.wait(), timeout=5.0)

    # sender_id should be transport_a's full UUID4 device_id (canonical form)
    assert sender_ids[0] == identity_a.device_id


@requires_liboqs
async def test_send_increments_bytes_sent(
    transport_a, transport_b, bundle_b
):
    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    transport_b.on_message(lambda s, m: received.set())

    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    b_device_id = list(transport_a.get_sessions().keys())[0]
    await transport_a.send(b_device_id, b"payload data here")
    await asyncio.wait_for(received.wait(), timeout=5.0)

    sess = transport_a.get_sessions()[b_device_id]
    assert sess.bytes_sent > 0


@requires_liboqs
async def test_send_returns_integer_msg_id(
    transport_a, transport_b, bundle_b
):
    addr_b = _addr_of(transport_b)
    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    b_device_id = list(transport_a.get_sessions().keys())[0]
    msg_id = await transport_a.send(b_device_id, b"hello")
    assert isinstance(msg_id, int)
    assert msg_id > 0


@requires_liboqs
async def test_bidirectional_message_exchange(
    transport_a, transport_b, bundle_b, identity_a, identity_b
):
    """Both nodes send to each other after handshake."""
    addr_b = _addr_of(transport_b)

    a_received = asyncio.Event()
    b_received = asyncio.Event()
    a_msgs = []
    b_msgs = []

    async def on_msg_a(sid, pt):
        a_msgs.append(pt)
        a_received.set()

    async def on_msg_b(sid, pt):
        b_msgs.append(pt)
        b_received.set()

    transport_a.on_message(on_msg_a)
    transport_b.on_message(on_msg_b)

    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.1)

    b_id = list(transport_a.get_sessions().keys())[0]
    a_id = list(transport_b.get_sessions().keys())[0]

    await transport_a.send(b_id, b"A->B message")
    await transport_b.send(a_id, b"B->A message")

    await asyncio.wait_for(b_received.wait(), timeout=5.0)
    await asyncio.wait_for(a_received.wait(), timeout=5.0)

    assert b"A->B message" in b_msgs
    assert b"B->A message" in a_msgs


# ---------------------------------------------------------------------------
# Group 10 — Fragmentation
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_large_message_fragmented_and_reassembled(
    transport_a, transport_b, bundle_b
):
    """Send a message larger than one packet; verify complete reassembly."""
    from mesh.transport import MAX_FRAG_PT

    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    messages = []

    async def on_msg(sid, pt):
        messages.append(pt)
        received.set()

    transport_b.on_message(on_msg)
    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    # Create a message that requires exactly 3 fragments
    large_msg = bytes(range(256)) * (MAX_FRAG_PT * 3 // 256 + 1)
    large_msg = large_msg[: MAX_FRAG_PT * 3 - 10]

    b_device_id = list(transport_a.get_sessions().keys())[0]
    await transport_a.send(b_device_id, large_msg)

    await asyncio.wait_for(received.wait(), timeout=10.0)
    assert messages[0] == large_msg


@requires_liboqs
async def test_fragment_total_zero_rejected(transport_a, transport_b, bundle_b):
    """Packet with frag_total=0 must be silently dropped."""
    from mesh.transport import (
        PacketType, build_packet, sender_id_bytes, FRAG_HEADER_SIZE, NONCE_SIZE
    )

    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    transport_b.on_message(lambda s, m: received.set())

    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    sess = list(transport_a.get_sessions().values())[0]

    # Build a malformed MESSAGE with frag_total=0
    msg_id = 0xDEAD
    bad_frag_header = struct.pack("!IBB", msg_id, 0, 0)  # frag_idx=0, frag_total=0
    nonce = os.urandom(12)
    ct = b"\x00" * 32
    payload = bad_frag_header + nonce + ct
    sid = sender_id_bytes(transport_a._identity.device_id)

    # Manually send the malformed packet
    pkt = build_packet(PacketType.MESSAGE, sid, payload, sess.session_key)
    transport_a._send_raw(pkt, _addr_of(transport_b))

    # Give it time; callback must NOT be called
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(received.wait(), timeout=0.5)


@requires_liboqs
async def test_fragment_idx_out_of_range_rejected(transport_a, transport_b, bundle_b):
    """Packet with frag_idx >= frag_total must be silently dropped."""
    from mesh.transport import (
        PacketType, build_packet, sender_id_bytes
    )

    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    transport_b.on_message(lambda s, m: received.set())

    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    sess = list(transport_a.get_sessions().values())[0]

    # frag_idx=5 but frag_total=3 → invalid
    bad_frag_header = struct.pack("!IBB", 0xBEEF, 5, 3)
    nonce = os.urandom(12)
    ct = b"\x00" * 32
    payload = bad_frag_header + nonce + ct
    sid = sender_id_bytes(transport_a._identity.device_id)

    pkt = build_packet(PacketType.MESSAGE, sid, payload, sess.session_key)
    transport_a._send_raw(pkt, _addr_of(transport_b))

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(received.wait(), timeout=0.5)


# ---------------------------------------------------------------------------
# Group 11 — ACK and retry
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_ack_removes_from_in_flight(transport_a, transport_b, bundle_b):
    """After receiving ACK, in-flight dict must be empty for that msg_id."""
    addr_b = _addr_of(transport_b)
    received = asyncio.Event()
    transport_b.on_message(lambda s, m: received.set())

    await asyncio.wait_for(transport_a.connect(addr_b, bundle_b), timeout=5.0)
    await asyncio.sleep(0.05)

    b_device_id = list(transport_a.get_sessions().keys())[0]
    msg_id = await transport_a.send(b_device_id, b"ack test")

    # Wait for message to be received and ACK to come back
    await asyncio.wait_for(received.wait(), timeout=5.0)
    await asyncio.sleep(0.1)  # allow ACK to propagate

    assert msg_id not in transport_a._in_flight


@requires_liboqs
async def test_retry_fires_send_failed_callback_after_max_retries(identity_a):
    """When no ACK is received, send_failed callback fires after MAX_RETRIES."""
    from mesh.transport import MeshTransport, RETRY_BACKOFFS

    t = MeshTransport(identity_a)
    await t.start("127.0.0.1", 0)

    failed_msg_ids = []
    t.on_send_failed(lambda mid, addr: failed_msg_ids.append(mid))

    # Inject a fake in-flight entry with a past deadline so retry fires immediately
    from mesh.transport import _InFlight
    fake_id = 9999
    # Build a minimal fake packet (won't be sent anywhere meaningful)
    dummy_pkt = b"\x01\x05" + b"a" * 32 + b"\x00\x04" + b"data" + b"\x00" * 16
    t._in_flight[fake_id] = _InFlight(
        msg_id=fake_id,
        packet_bytes=dummy_pkt,
        peer_addr=("127.0.0.1", 1),
        attempt=3,           # already exhausted
        next_retry_at=0.0,   # overdue
    )

    # Give retry loop one cycle
    await asyncio.sleep(0.2)

    assert fake_id in failed_msg_ids
    assert fake_id not in t._in_flight
    await t.stop()


@requires_liboqs
async def test_retry_loop_retransmits_before_giving_up(identity_a):
    """Verify retry loop re-sends a packet at least once before marking failed."""
    from mesh.transport import MeshTransport, _InFlight

    t = MeshTransport(identity_a)
    await t.start("127.0.0.1", 0)

    send_calls = []
    original_send_raw = t._send_raw
    t._send_raw = lambda pkt, addr: send_calls.append((pkt, addr))

    fake_id = 8888
    dummy_pkt = b"\x01\x05" + b"a" * 32 + b"\x00\x04" + b"data" + b"\x00" * 16
    t._in_flight[fake_id] = _InFlight(
        msg_id=fake_id,
        packet_bytes=dummy_pkt,
        peer_addr=("127.0.0.1", 1),
        attempt=0,
        next_retry_at=0.0,  # due immediately
    )

    await asyncio.sleep(0.3)  # allow a couple retry cycles

    # Should have retransmitted at least once
    assert len(send_calls) >= 1
    t._send_raw = original_send_raw
    await t.stop()
