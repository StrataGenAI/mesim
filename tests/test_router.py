"""
tests/test_router.py — Unit tests for mesh/router.py

Groups:
  1. Constants
  2. RouteEntry dataclass
  3. update_route
  4. route_to
  5. get_route_table
  6. purge_stale
  7. Originator packet wire format
  8. Originator signature verification
  9. Forwarding rules (hop_count, loop prevention)
 10. MeshRouter lifecycle and integration
"""

import asyncio
import os
import struct
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.crypto import sign_message, verify_signature
from core.identity import Rank, create_identity, get_public_bundle
from mesh.router import (
    MAX_HOPS,
    ORIGINATOR_INTERVAL,
    STALE_ROUTE_SECONDS,
    MeshRouter,
    RouteEntry,
    pack_originator,
    unpack_originator,
)
from mesh.transport import PacketType

requires_liboqs = pytest.mark.skipif(
    not __import__("core.crypto", fromlist=["_oqs_available"])._oqs_available,
    reason="liboqs not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_a():
    return create_identity("ALPHA", Rank.NCO)


@pytest.fixture
def identity_b():
    return create_identity("BRAVO", Rank.NCO)


def _make_transport(sessions: dict | None = None):
    t = MagicMock()
    t.get_sessions.return_value = sessions or {}
    t._sender_id = b"a" * 32
    t._sessions_by_addr = {}
    t._send_raw = MagicMock()
    t.on_raw_packet = MagicMock()
    return t


# ---------------------------------------------------------------------------
# Group 1 — Constants
# ---------------------------------------------------------------------------


def test_max_hops_is_7():
    assert MAX_HOPS == 7


def test_originator_interval_is_10():
    assert ORIGINATOR_INTERVAL == 10.0


def test_stale_route_seconds_is_60():
    assert STALE_ROUTE_SECONDS == 60


# ---------------------------------------------------------------------------
# Group 2 — RouteEntry dataclass
# ---------------------------------------------------------------------------


def test_route_entry_fields():
    now = time.time()
    entry = RouteEntry(
        originator_id="abc123",
        next_hop_id="hop1",
        next_hop_addr=("10.0.0.1", 7777),
        hop_count=2,
        last_seen=now,
        seq_num=5,
    )
    assert entry.originator_id == "abc123"
    assert entry.next_hop_id == "hop1"
    assert entry.next_hop_addr == ("10.0.0.1", 7777)
    assert entry.hop_count == 2
    assert entry.last_seen == now
    assert entry.seq_num == 5


# ---------------------------------------------------------------------------
# Group 3 — update_route
# ---------------------------------------------------------------------------


def test_update_route_stores_entry(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    router.update_route("peer1", "hop1", ("10.0.0.1", 7), 1, seq_num=1, timestamp=time.time())
    assert "peer1" in router.get_route_table()


def test_update_route_replaces_lower_seq_num(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    now = time.time()
    router.update_route("peer1", "hop_old", ("10.0.0.1", 7), 3, seq_num=1, timestamp=now)
    router.update_route("peer1", "hop_new", ("10.0.0.2", 7), 2, seq_num=2, timestamp=now)
    entry = router.get_route_table()["peer1"]
    assert entry.next_hop_id == "hop_new"
    assert entry.seq_num == 2


def test_update_route_ignores_older_seq_num(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    now = time.time()
    router.update_route("peer1", "hop_new", ("10.0.0.2", 7), 2, seq_num=5, timestamp=now)
    router.update_route("peer1", "hop_old", ("10.0.0.1", 7), 1, seq_num=3, timestamp=now)
    entry = router.get_route_table()["peer1"]
    assert entry.next_hop_id == "hop_new"
    assert entry.seq_num == 5


def test_update_route_prefers_lower_hop_count_on_equal_seq(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    now = time.time()
    router.update_route("peer1", "hop_far",  ("10.0.0.1", 7), 4, seq_num=1, timestamp=now)
    router.update_route("peer1", "hop_near", ("10.0.0.2", 7), 2, seq_num=1, timestamp=now)
    entry = router.get_route_table()["peer1"]
    assert entry.hop_count == 2
    assert entry.next_hop_id == "hop_near"


# ---------------------------------------------------------------------------
# Group 4 — route_to
# ---------------------------------------------------------------------------


def test_route_to_unknown_peer_returns_none(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    assert router.route_to("nobody") is None


def test_route_to_returns_entry(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    router.update_route("peer1", "hop1", ("1.2.3.4", 7), 1, seq_num=1, timestamp=time.time())
    entry = router.route_to("peer1")
    assert entry is not None
    assert entry.next_hop_id == "hop1"


def test_route_to_ignores_stale_route(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    stale_ts = time.time() - STALE_ROUTE_SECONDS - 1
    router.update_route("peer1", "stale_hop", ("1.2.3.4", 7), 1, seq_num=1, timestamp=stale_ts)
    assert router.route_to("peer1") is None


def test_route_to_prefers_lower_hop_count(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    now = time.time()
    router.update_route("peer1", "far",  ("1.2.3.4", 7), 4, seq_num=1, timestamp=now)
    router.update_route("peer1", "near", ("1.2.3.5", 7), 2, seq_num=2, timestamp=now)
    assert router.route_to("peer1").hop_count == 2


def test_route_to_fresh_after_stale(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    stale_ts = time.time() - STALE_ROUTE_SECONDS - 1
    router.update_route("peer1", "stale", ("1.0.0.1", 7), 1, seq_num=1, timestamp=stale_ts)
    # Now update with fresh timestamp
    router.update_route("peer1", "fresh", ("1.0.0.2", 7), 1, seq_num=2, timestamp=time.time())
    assert router.route_to("peer1") is not None
    assert router.route_to("peer1").next_hop_id == "fresh"


# ---------------------------------------------------------------------------
# Group 5 — get_route_table
# ---------------------------------------------------------------------------


def test_get_route_table_returns_snapshot(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    router.update_route("peer1", "hop1", ("1.2.3.4", 7), 1, seq_num=1, timestamp=time.time())
    table = router.get_route_table()
    # Mutating the snapshot must not affect the router
    table["peer1"] = RouteEntry("peer1", "fake", ("0.0.0.0", 0), 99, 0.0, 99)
    assert router.get_route_table()["peer1"].next_hop_id == "hop1"


def test_get_route_table_empty(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    assert router.get_route_table() == {}


# ---------------------------------------------------------------------------
# Group 6 — purge_stale
# ---------------------------------------------------------------------------


def test_purge_stale_removes_old(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    stale_ts = time.time() - STALE_ROUTE_SECONDS - 1
    router.update_route("peer1", "hop1", ("1.2.3.4", 7), 1, seq_num=1, timestamp=stale_ts)
    removed = router.purge_stale()
    assert removed == 1
    assert "peer1" not in router.get_route_table()


def test_purge_stale_keeps_fresh(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    router.update_route("peer1", "hop1", ("1.2.3.4", 7), 1, seq_num=1, timestamp=time.time())
    assert router.purge_stale() == 0
    assert "peer1" in router.get_route_table()


def test_purge_stale_returns_count(identity_a):
    router = MeshRouter(identity_a, _make_transport())
    stale_ts = time.time() - STALE_ROUTE_SECONDS - 1
    for i in range(3):
        router.update_route(f"peer{i}", "hop1", ("1.2.3.4", 7), 1, seq_num=1, timestamp=stale_ts)
    assert router.purge_stale() == 3


# ---------------------------------------------------------------------------
# Group 7 — Originator packet wire format
# ---------------------------------------------------------------------------


@requires_liboqs
def test_pack_originator_length(identity_a):
    bundle_a = get_public_bundle(identity_a)
    data = pack_originator(
        originator_id=identity_a.device_id,
        seq_num=1,
        hop_count=0,
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_a.signing_keypair.signing_key,
    )
    # 32B originator_id + 4B seq + 1B hop + 1B max_hops + 8B ts + 64B sig = 110 bytes
    assert len(data) == 110


@requires_liboqs
def test_pack_unpack_originator_roundtrip(identity_a):
    ts = time.time()
    data = pack_originator(
        originator_id=identity_a.device_id,
        seq_num=42,
        hop_count=2,
        max_hops=MAX_HOPS,
        timestamp=ts,
        signing_key=identity_a.signing_keypair.signing_key,
    )
    orig_id, seq_num, hop_count, max_hops, timestamp, sig = unpack_originator(data)
    assert seq_num == 42
    assert hop_count == 2
    assert max_hops == MAX_HOPS
    assert abs(timestamp - ts) < 0.001


# ---------------------------------------------------------------------------
# Group 8 — Originator signature verification
# ---------------------------------------------------------------------------


@requires_liboqs
def test_originator_signature_valid(identity_a):
    bundle_a = get_public_bundle(identity_a)
    data = pack_originator(
        originator_id=identity_a.device_id,
        seq_num=1,
        hop_count=0,
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_a.signing_keypair.signing_key,
    )
    orig_id, seq_num, hop_count, max_hops, ts, sig = unpack_originator(data)
    # Verify sig over the fields that don't change per-hop
    from mesh.router import _originator_signed_bytes
    signed = _originator_signed_bytes(orig_id, seq_num, max_hops, ts)
    verify_signature(signed, sig, bundle_a.verify_key)  # must not raise


@requires_liboqs
def test_originator_bad_signature_raises(identity_a, identity_b):
    bundle_b = get_public_bundle(identity_b)
    data = pack_originator(
        originator_id=identity_a.device_id,
        seq_num=1,
        hop_count=0,
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_a.signing_keypair.signing_key,
    )
    orig_id, seq_num, hop_count, max_hops, ts, sig = unpack_originator(data)
    from mesh.router import _originator_signed_bytes
    from cryptography.exceptions import InvalidSignature
    signed = _originator_signed_bytes(orig_id, seq_num, max_hops, ts)
    with pytest.raises(InvalidSignature):
        verify_signature(signed, sig, bundle_b.verify_key)  # wrong key


# ---------------------------------------------------------------------------
# Group 9 — Forwarding rules
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_handle_originator_updates_route_table(identity_a, identity_b):
    bundle_b = get_public_bundle(identity_b)
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    # Teach router about identity_b so it can verify the sig
    router._known_bundles[identity_b.device_id] = bundle_b

    from mesh.transport import sender_id_bytes
    sender_id = sender_id_bytes(identity_b.device_id)
    data = pack_originator(
        originator_id=identity_b.device_id,
        seq_num=1,
        hop_count=0,
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_b.signing_keypair.signing_key,
    )
    from_addr = ("10.0.0.2", 7777)
    await router._handle_originator(data, from_addr, sender_id)

    table = router.get_route_table()
    orig_id_key = identity_b.device_id.replace("-", "")[:32]
    assert orig_id_key in table or identity_b.device_id in table


@requires_liboqs
async def test_handle_originator_does_not_forward_at_max_hops(identity_a, identity_b):
    bundle_b = get_public_bundle(identity_b)
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    router._known_bundles[identity_b.device_id] = bundle_b

    from mesh.transport import sender_id_bytes
    sender_id = sender_id_bytes(identity_b.device_id)
    data = pack_originator(
        originator_id=identity_b.device_id,
        seq_num=1,
        hop_count=MAX_HOPS,   # already at max
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_b.signing_keypair.signing_key,
    )
    await router._handle_originator(data, ("10.0.0.2", 7777), sender_id)
    transport._send_raw.assert_not_called()


@requires_liboqs
async def test_handle_originator_forwards_below_max_hops(identity_a, identity_b):
    bundle_b = get_public_bundle(identity_b)
    session = MagicMock()
    session.peer_addr = ("10.0.0.3", 7777)
    transport = _make_transport(sessions={"peer3": session})
    router = MeshRouter(identity_a, transport)
    router._known_bundles[identity_b.device_id] = bundle_b

    from mesh.transport import sender_id_bytes
    sender_id = sender_id_bytes(identity_b.device_id)
    from_addr = ("10.0.0.2", 7777)
    data = pack_originator(
        originator_id=identity_b.device_id,
        seq_num=1,
        hop_count=2,   # below max
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_b.signing_keypair.signing_key,
    )
    await router._handle_originator(data, from_addr, sender_id)
    # Should forward to all sessions except the incoming one
    transport._send_raw.assert_called()


@requires_liboqs
async def test_handle_originator_no_duplicate_seq(identity_a, identity_b):
    bundle_b = get_public_bundle(identity_b)
    session = MagicMock()
    session.peer_addr = ("10.0.0.3", 7777)
    transport = _make_transport(sessions={"peer3": session})
    router = MeshRouter(identity_a, transport)
    router._known_bundles[identity_b.device_id] = bundle_b

    from mesh.transport import sender_id_bytes
    sender_id = sender_id_bytes(identity_b.device_id)
    data = pack_originator(
        originator_id=identity_b.device_id,
        seq_num=1,
        hop_count=1,
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_b.signing_keypair.signing_key,
    )
    await router._handle_originator(data, ("10.0.0.2", 7777), sender_id)
    call_count_after_first = transport._send_raw.call_count
    # Same seq_num again — must not re-forward
    await router._handle_originator(data, ("10.0.0.2", 7777), sender_id)
    assert transport._send_raw.call_count == call_count_after_first


@requires_liboqs
async def test_handle_originator_self_not_forwarded(identity_a):
    """Originator from ourself must be silently ignored."""
    session = MagicMock()
    session.peer_addr = ("10.0.0.3", 7777)
    transport = _make_transport(sessions={"peer3": session})
    transport._sender_id = __import__("mesh.transport", fromlist=["sender_id_bytes"]).sender_id_bytes(identity_a.device_id)
    router = MeshRouter(identity_a, transport)

    from mesh.transport import sender_id_bytes
    our_sender_id = sender_id_bytes(identity_a.device_id)
    data = pack_originator(
        originator_id=identity_a.device_id,
        seq_num=1,
        hop_count=0,
        max_hops=MAX_HOPS,
        timestamp=time.time(),
        signing_key=identity_a.signing_keypair.signing_key,
    )
    await router._handle_originator(data, ("127.0.0.1", 7777), our_sender_id)
    transport._send_raw.assert_not_called()


# ---------------------------------------------------------------------------
# Group 10 — MeshRouter lifecycle and integration
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_router_start_registers_on_raw_packet(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    await router.start()
    transport.on_raw_packet.assert_called_once_with(
        PacketType.ORIGINATOR, router._handle_originator
    )
    await router.stop()


@requires_liboqs
async def test_router_stop_cancels_task(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    await router.start()
    assert router._originator_task is not None
    await router.stop()
    assert router._originator_task is None


@requires_liboqs
async def test_router_broadcast_originator_sends_to_sessions(identity_a):
    session = MagicMock()
    session.peer_addr = ("10.0.0.1", 7777)
    transport = _make_transport(sessions={"peer1": session})
    router = MeshRouter(identity_a, transport)
    await router._broadcast_originator()
    transport._send_raw.assert_called()


@requires_liboqs
async def test_router_originator_seq_num_increments(identity_a):
    transport = _make_transport()
    router = MeshRouter(identity_a, transport)
    seq1 = router._seq_num
    await router._broadcast_originator()
    assert router._seq_num == seq1 + 1
