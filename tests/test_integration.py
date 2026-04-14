"""
tests/test_integration.py — Two-node end-to-end integration tests

Spins up two real MeshTransport instances on loopback UDP ports in the same
asyncio event loop.  Tests the complete path:

  Test 1 (test_two_node_handshake_and_live_delivery):
    - Node A connects to Node B (full 4-step hybrid KEM handshake)
    - A sends 5 messages → B receives all 5, order preserved

  Test 2 (test_store_forward_offline_queue_then_reconnect):
    - 3 messages queued in A's ForwardQueue while B has no session
    - B connects to A → on_peer_connected fires → FlushPeer delivers all 3
    - All 3 received by B, order preserved, no duplicates

  Test 3 (test_full_flow_live_then_offline_then_reconnect):
    - A↔B handshake → 5 live messages
    - B disconnects (transport stopped)
    - A queues 3 more messages in ForwardQueue directly
    - New B transport starts, A connects to new B
    - on_peer_connected fires → flush delivers 3 queued messages
    - B receives all 8 total, in order

Requires liboqs.  All tests use port=0 (OS-assigned free port).
"""

from __future__ import annotations

import asyncio
import os

import pytest

from core.crypto import SessionKey, _oqs_available
from core.identity import Rank, create_identity, get_public_bundle
from mesh.store_forward import ForwardQueue, StoreForward
from mesh.transport import MeshTransport

requires_liboqs = pytest.mark.skipif(
    not _oqs_available, reason="liboqs not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr_of(transport: MeshTransport) -> tuple[str, int]:
    return transport._udp_transport.get_extra_info("sockname")


# ---------------------------------------------------------------------------
# Test 1 — handshake + 5 live messages
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_two_node_handshake_and_live_delivery(tmp_path):
    """Full 4-step KEM handshake; A sends 5 messages, B receives all 5 in order."""
    identity_a = create_identity("ALPHA-1", Rank.COMMAND)
    identity_b = create_identity("BRAVO-1", Rank.SQUAD)

    transport_a = MeshTransport(identity_a)
    transport_b = MeshTransport(identity_b)
    await transport_a.start("127.0.0.1", 0)
    await transport_b.start("127.0.0.1", 0)

    received: list[bytes] = []
    all_received = asyncio.Event()
    EXPECTED = 5

    async def on_b_message(peer_id: str, payload: bytes) -> None:
        received.append(payload)
        if len(received) >= EXPECTED:
            all_received.set()

    transport_b.on_message(on_b_message)

    try:
        bundle_b = get_public_bundle(identity_b)
        ok = await transport_a.connect(_addr_of(transport_b), bundle_b)
        assert ok, "Handshake A→B failed"

        sessions = transport_a.get_sessions()
        assert len(sessions) == 1
        b_id = list(sessions.keys())[0]
        assert b_id == identity_b.device_id

        messages = [f"live-{i}".encode() for i in range(EXPECTED)]
        for msg in messages:
            await transport_a.send(b_id, msg)

        await asyncio.wait_for(all_received.wait(), timeout=10.0)

        assert len(received) == EXPECTED
        assert received == messages  # order preserved

    finally:
        await transport_a.stop()
        await transport_b.stop()


# ---------------------------------------------------------------------------
# Test 2 — store-and-forward: queue while offline, deliver on reconnect
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_store_forward_offline_queue_then_reconnect(tmp_path):
    """3 messages queued while B has no session; delivered when B connects to A."""
    identity_a = create_identity("CHARLIE-2", Rank.NCO)
    identity_b = create_identity("DELTA-2", Rank.SQUAD)

    transport_a = MeshTransport(identity_a)
    transport_b = MeshTransport(identity_b)
    await transport_a.start("127.0.0.1", 0)
    await transport_b.start("127.0.0.1", 0)

    fwd_db = str(tmp_path / "charlie.fwd.db")
    store_key = SessionKey(raw=os.urandom(32))
    fwd_queue = ForwardQueue(fwd_db, store_key)
    fwd_queue.open()
    sf_a = StoreForward(fwd_queue, transport_a)
    await sf_a.start()

    received: list[bytes] = []
    all_received = asyncio.Event()
    EXPECTED = 3

    async def on_b_message(peer_id: str, payload: bytes) -> None:
        received.append(payload)
        if len(received) >= EXPECTED:
            all_received.set()

    transport_b.on_message(on_b_message)

    try:
        b_id = identity_b.device_id
        messages = [f"queued-{i}".encode() for i in range(EXPECTED)]

        # No session exists yet — messages go straight to the queue
        for msg in messages:
            queued = await sf_a.send_or_queue(b_id, msg)
            assert queued is False, "Expected messages to be queued (no session)"

        assert fwd_queue.queue_size(b_id) == EXPECTED

        # B initiates the connection; on_peer_connected fires in sf_a → flush
        bundle_a = get_public_bundle(identity_a)
        ok = await transport_b.connect(_addr_of(transport_a), bundle_a)
        assert ok, "Handshake B→A failed"

        await asyncio.wait_for(all_received.wait(), timeout=10.0)

        assert len(received) == EXPECTED
        assert received == messages          # order preserved
        assert fwd_queue.queue_size(b_id) == 0  # queue drained

    finally:
        await sf_a.stop()
        fwd_queue.close()
        await transport_a.stop()
        await transport_b.stop()


# ---------------------------------------------------------------------------
# Test 3 — full flow: 5 live + B offline + 3 queued + reconnect = 8 total
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_full_flow_live_then_offline_then_reconnect(tmp_path):
    """
    A↔B: 5 live messages.
    B stops.  A queues 3 more (no live session).
    New B starts.  A connects → flush → B receives 3.
    Total delivered to B across both phases: 8 messages.
    """
    identity_a = create_identity("ECHO-3", Rank.COMMAND)
    identity_b = create_identity("FOXTROT-3", Rank.SQUAD)

    # ── Phase 1: live delivery ──────────────────────────────────────────────
    transport_a = MeshTransport(identity_a)
    transport_b1 = MeshTransport(identity_b)
    await transport_a.start("127.0.0.1", 0)
    await transport_b1.start("127.0.0.1", 0)

    fwd_db = str(tmp_path / "echo.fwd.db")
    store_key = SessionKey(raw=os.urandom(32))
    fwd_queue = ForwardQueue(fwd_db, store_key)
    fwd_queue.open()
    sf_a = StoreForward(fwd_queue, transport_a)
    await sf_a.start()

    b_id = identity_b.device_id
    phase1_received: list[bytes] = []
    phase1_done = asyncio.Event()

    async def on_b1_message(peer_id: str, payload: bytes) -> None:
        phase1_received.append(payload)
        if len(phase1_received) >= 5:
            phase1_done.set()

    transport_b1.on_message(on_b1_message)

    bundle_b = get_public_bundle(identity_b)
    ok = await transport_a.connect(_addr_of(transport_b1), bundle_b)
    assert ok, "Phase-1 handshake failed"

    live_messages = [f"live-{i}".encode() for i in range(5)]
    for msg in live_messages:
        await transport_a.send(b_id, msg)

    await asyncio.wait_for(phase1_done.wait(), timeout=10.0)
    assert phase1_received == live_messages

    # ── Phase 2: B goes offline; A queues 3 ────────────────────────────────
    await transport_b1.stop()

    queued_messages = [f"queued-{i}".encode() for i in range(3)]
    for msg in queued_messages:
        fwd_queue.enqueue(b_id, msg)

    assert fwd_queue.queue_size(b_id) == 3

    # ── Phase 3: new B comes online, A reconnects ───────────────────────────
    transport_b2 = MeshTransport(identity_b)   # same identity, fresh transport
    await transport_b2.start("127.0.0.1", 0)

    phase2_received: list[bytes] = []
    phase2_done = asyncio.Event()

    async def on_b2_message(peer_id: str, payload: bytes) -> None:
        phase2_received.append(payload)
        if len(phase2_received) >= 3:
            phase2_done.set()

    transport_b2.on_message(on_b2_message)

    ok2 = await transport_a.connect(_addr_of(transport_b2), bundle_b)
    assert ok2, "Phase-3 handshake failed"

    # on_peer_connected fires for b_id in sf_a → flush 3 queued messages
    await asyncio.wait_for(phase2_done.wait(), timeout=10.0)

    assert phase2_received == queued_messages   # order preserved
    assert fwd_queue.queue_size(b_id) == 0      # queue drained

    # Total across both phases: 8 messages, no duplicates
    all_received = phase1_received + phase2_received
    assert len(all_received) == 8
    assert len(set(all_received)) == 8          # no duplicates

    await sf_a.stop()
    fwd_queue.close()
    await transport_a.stop()
    await transport_b2.stop()
