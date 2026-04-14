"""
tests/test_api.py — MESIM FastAPI device server tests

All subsystems are replaced with lightweight mock objects so tests run
without networking, real crypto, or a live database.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.server import SendResponse, create_app
from core.identity import Rank


# ---------------------------------------------------------------------------
# Helpers — minimal stub objects
# ---------------------------------------------------------------------------


def _make_identity(
    device_id: str = "aabbcc00" * 4,
    callsign: str = "ALPHA-1",
    rank: Rank = Rank.NCO,
) -> MagicMock:
    m = MagicMock()
    m.device_id = device_id
    m.callsign = callsign
    m.rank = rank
    return m


def _make_bundle(callsign: str = "BRAVO-2", rank: Rank = Rank.SQUAD) -> MagicMock:
    b = MagicMock()
    b.callsign = callsign
    b.rank = rank
    return b


def _make_session(
    peer_id: str = "ff00" * 8,
    host: str = "192.168.1.2",
    port: int = 7777,
    callsign: str = "BRAVO-2",
    rank: Rank = Rank.SQUAD,
    established_at: float = 1000.0,
    last_active: float = 1001.0,
    bytes_sent: int = 512,
    bytes_recv: int = 256,
) -> MagicMock:
    s = MagicMock()
    s.peer_id = peer_id
    s.peer_addr = (host, port)
    s.peer_bundle = _make_bundle(callsign, rank)
    s.established_at = established_at
    s.last_active = last_active
    s.bytes_sent = bytes_sent
    s.bytes_recv = bytes_recv
    return s


def _make_route(
    originator_id: str = "cc00" * 8,
    next_hop_id: str = "ff00" * 8,
    hop_count: int = 2,
    seq_num: int = 42,
    host: str = "192.168.1.2",
    port: int = 7777,
    last_seen: float = 1000.0,
) -> MagicMock:
    e = MagicMock()
    e.originator_id = originator_id
    e.next_hop_id = next_hop_id
    e.next_hop_addr = (host, port)
    e.hop_count = hop_count
    e.seq_num = seq_num
    e.last_seen = last_seen
    return e


def _make_stored_message(
    msg_id: int = 1,
    peer_id: str = "ff00" * 8,
    direction: str = "recv",
    plaintext: bytes = b"hello",
    timestamp: float = 1000.0,
    delivered: bool = False,
) -> MagicMock:
    m = MagicMock()
    m.id = msg_id
    m.peer_id = peer_id
    m.direction = direction
    m.plaintext = plaintext
    m.timestamp = timestamp
    m.delivered = delivered
    return m


def _build_client(
    *,
    identity=None,
    sessions: dict | None = None,
    routes: dict | None = None,
    send_or_queue_result: bool = True,
    send_or_queue_raises: Exception | None = None,
    save_message_result: int = 1,
    history: list | None = None,
    requests_per_minute: int = 100,
) -> TestClient:
    """Build a TestClient with fully mocked subsystems."""
    ident = identity or _make_identity()
    sessions = sessions if sessions is not None else {}
    routes = routes if routes is not None else {}
    history = history if history is not None else []

    transport = MagicMock()
    transport.get_sessions.return_value = sessions

    router = MagicMock()
    router.get_route_table.return_value = routes

    sf = MagicMock()
    if send_or_queue_raises is not None:
        sf.send_or_queue = AsyncMock(side_effect=send_or_queue_raises)
    else:
        sf.send_or_queue = AsyncMock(return_value=send_or_queue_result)

    ms = MagicMock()
    ms.save_message.return_value = save_message_result
    ms.get_history.return_value = history

    app = create_app(ident, transport, router, sf, ms,
                     requests_per_minute=requests_per_minute)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_status_ok(self) -> None:
        client = _build_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_returns_device_id(self) -> None:
        ident = _make_identity(device_id="deadbeef" * 4)
        client = _build_client(identity=ident)
        resp = client.get("/health")
        assert resp.json()["device_id"] == "deadbeef" * 4

    def test_returns_callsign(self) -> None:
        ident = _make_identity(callsign="DELTA-4")
        client = _build_client(identity=ident)
        resp = client.get("/health")
        assert resp.json()["callsign"] == "DELTA-4"

    def test_returns_rank_int(self) -> None:
        ident = _make_identity(rank=Rank.COMMAND)
        client = _build_client(identity=ident)
        resp = client.get("/health")
        assert resp.json()["rank"] == int(Rank.COMMAND)

    def test_returns_rank_name(self) -> None:
        ident = _make_identity(rank=Rank.OFFICER)
        client = _build_client(identity=ident)
        resp = client.get("/health")
        assert resp.json()["rank_name"] == "OFFICER"

    def test_returns_timestamp(self) -> None:
        before = time.time()
        client = _build_client()
        resp = client.get("/health")
        after = time.time()
        ts = resp.json()["timestamp"]
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# GET /peers
# ---------------------------------------------------------------------------


class TestPeers:
    def test_empty_returns_list(self) -> None:
        client = _build_client(sessions={})
        resp = client.get("/peers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_single_peer(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, callsign="BRAVO-2", rank=Rank.SQUAD)
        client = _build_client(sessions={pid: s})
        resp = client.get("/peers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["peer_id"] == pid
        assert data[0]["callsign"] == "BRAVO-2"

    def test_peer_rank_int(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, rank=Rank.NCO)
        client = _build_client(sessions={pid: s})
        resp = client.get("/peers")
        assert resp.json()[0]["rank"] == int(Rank.NCO)

    def test_peer_rank_name(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, rank=Rank.COMMAND)
        client = _build_client(sessions={pid: s})
        resp = client.get("/peers")
        assert resp.json()[0]["rank_name"] == "COMMAND"

    def test_peer_address(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, host="10.0.0.5", port=9999)
        client = _build_client(sessions={pid: s})
        resp = client.get("/peers")
        assert resp.json()[0]["addr_host"] == "10.0.0.5"
        assert resp.json()[0]["addr_port"] == 9999

    def test_peer_bytes_stats(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, bytes_sent=1024, bytes_recv=2048)
        client = _build_client(sessions={pid: s})
        resp = client.get("/peers")
        assert resp.json()[0]["bytes_sent"] == 1024
        assert resp.json()[0]["bytes_recv"] == 2048

    def test_multiple_peers(self) -> None:
        s1 = _make_session(peer_id="aa" * 16, callsign="ALPHA-1")
        s2 = _make_session(peer_id="bb" * 16, callsign="BRAVO-2")
        client = _build_client(sessions={"aa" * 16: s1, "bb" * 16: s2})
        resp = client.get("/peers")
        callsigns = {p["callsign"] for p in resp.json()}
        assert callsigns == {"ALPHA-1", "BRAVO-2"}

    def test_established_at_field(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, established_at=12345.0)
        client = _build_client(sessions={pid: s})
        resp = client.get("/peers")
        assert resp.json()[0]["established_at"] == 12345.0


# ---------------------------------------------------------------------------
# GET /routes
# ---------------------------------------------------------------------------


class TestRoutes:
    def test_empty_returns_list(self) -> None:
        client = _build_client(routes={})
        resp = client.get("/routes")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_single_route(self) -> None:
        oid = "cc00" * 8
        e = _make_route(originator_id=oid, hop_count=3, seq_num=7)
        client = _build_client(routes={oid: e})
        resp = client.get("/routes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["originator_id"] == oid
        assert data[0]["hop_count"] == 3
        assert data[0]["seq_num"] == 7

    def test_route_next_hop(self) -> None:
        oid = "cc00" * 8
        nhid = "ff00" * 8
        e = _make_route(originator_id=oid, next_hop_id=nhid)
        client = _build_client(routes={oid: e})
        resp = client.get("/routes")
        assert resp.json()[0]["next_hop_id"] == nhid

    def test_route_address(self) -> None:
        oid = "cc00" * 8
        e = _make_route(originator_id=oid, host="10.0.0.5", port=7778)
        client = _build_client(routes={oid: e})
        resp = client.get("/routes")
        assert resp.json()[0]["next_hop_host"] == "10.0.0.5"
        assert resp.json()[0]["next_hop_port"] == 7778

    def test_route_last_seen(self) -> None:
        oid = "cc00" * 8
        e = _make_route(originator_id=oid, last_seen=9999.0)
        client = _build_client(routes={oid: e})
        resp = client.get("/routes")
        assert resp.json()[0]["last_seen"] == 9999.0

    def test_multiple_routes(self) -> None:
        oid1, oid2 = "aa" * 16, "bb" * 16
        e1 = _make_route(originator_id=oid1, hop_count=1)
        e2 = _make_route(originator_id=oid2, hop_count=2)
        client = _build_client(routes={oid1: e1, oid2: e2})
        resp = client.get("/routes")
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# POST /send
# ---------------------------------------------------------------------------


class TestSend:
    def test_immediate_delivery(self) -> None:
        client = _build_client(send_or_queue_result=True, save_message_result=5)
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["queued"] is False
        assert body["msg_id"] == 5

    def test_queued_delivery(self) -> None:
        client = _build_client(send_or_queue_result=False, save_message_result=3)
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "hello"},
        )
        assert resp.status_code == 200
        assert resp.json()["queued"] is True

    def test_returns_peer_id(self) -> None:
        pid = "ff00" * 8
        client = _build_client()
        resp = client.post("/send", json={"peer_id": pid, "message": "x"})
        assert resp.json()["peer_id"] == pid

    def test_transport_error_returns_503(self) -> None:
        client = _build_client(send_or_queue_raises=RuntimeError("queue full"))
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "hello"},
        )
        assert resp.status_code == 503

    def test_503_detail_contains_error(self) -> None:
        client = _build_client(send_or_queue_raises=OverflowError("max peers"))
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "hello"},
        )
        assert "max peers" in resp.json()["detail"]

    def test_missing_peer_id_returns_422(self) -> None:
        client = _build_client()
        resp = client.post("/send", json={"message": "hello"})
        assert resp.status_code == 422

    def test_missing_message_returns_422(self) -> None:
        client = _build_client()
        resp = client.post("/send", json={"peer_id": "ff00" * 8})
        assert resp.status_code == 422

    def test_saves_to_message_store(self) -> None:
        transport = MagicMock()
        transport.get_sessions.return_value = {}
        sf = MagicMock()
        sf.send_or_queue = AsyncMock(return_value=True)
        ms = MagicMock()
        ms.save_message.return_value = 1
        ms.get_history.return_value = []
        app = create_app(_make_identity(), transport, MagicMock(), sf, ms)
        client = TestClient(app)
        client.post("/send", json={"peer_id": "ff00" * 8, "message": "stored"})
        ms.save_message.assert_called_once_with("ff00" * 8, "sent", b"stored")


# ---------------------------------------------------------------------------
# GET /messages/{peer_id}
# ---------------------------------------------------------------------------


class TestMessages:
    def test_empty_history(self) -> None:
        client = _build_client(history=[])
        resp = client.get("/messages/ff00ff00")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_single_message(self) -> None:
        pid = "ff00" * 8
        msg = _make_stored_message(msg_id=1, plaintext=b"hello", direction="recv")
        client = _build_client(history=[msg])
        resp = client.get(f"/messages/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["plaintext"] == "hello"
        assert data[0]["id"] == 1

    def test_direction_sent(self) -> None:
        msg = _make_stored_message(direction="sent", plaintext=b"out")
        client = _build_client(history=[msg])
        resp = client.get("/messages/any")
        assert resp.json()[0]["direction"] == "sent"

    def test_direction_recv(self) -> None:
        msg = _make_stored_message(direction="recv", plaintext=b"in")
        client = _build_client(history=[msg])
        resp = client.get("/messages/any")
        assert resp.json()[0]["direction"] == "recv"

    def test_delivered_flag(self) -> None:
        msg = _make_stored_message(delivered=True)
        client = _build_client(history=[msg])
        resp = client.get("/messages/any")
        assert resp.json()[0]["delivered"] is True

    def test_timestamp_preserved(self) -> None:
        msg = _make_stored_message(timestamp=12345.678)
        client = _build_client(history=[msg])
        resp = client.get("/messages/any")
        assert resp.json()[0]["timestamp"] == 12345.678

    def test_utf8_encoding_label(self) -> None:
        msg = _make_stored_message(plaintext=b"text")
        client = _build_client(history=[msg])
        resp = client.get("/messages/any")
        assert resp.json()[0]["encoding"] == "utf-8"

    def test_binary_payload_base64(self) -> None:
        import base64

        raw = b"\xff\xfe\xfd\xfc"  # guaranteed invalid UTF-8
        msg = _make_stored_message(plaintext=raw)
        client = _build_client(history=[msg])
        resp = client.get("/messages/any")
        item = resp.json()[0]
        assert item["encoding"] == "base64"
        assert item["plaintext"] == base64.b64encode(raw).decode("ascii")

    def test_limit_capped_at_500(self) -> None:
        transport = MagicMock()
        transport.get_sessions.return_value = {}
        ms = MagicMock()
        ms.get_history.return_value = []
        app = create_app(_make_identity(), transport, MagicMock(), MagicMock(), ms)
        client = TestClient(app)
        client.get("/messages/peer?limit=9999")
        ms.get_history.assert_called_once_with("peer", limit=500)

    def test_default_limit_100(self) -> None:
        transport = MagicMock()
        transport.get_sessions.return_value = {}
        ms = MagicMock()
        ms.get_history.return_value = []
        app = create_app(_make_identity(), transport, MagicMock(), MagicMock(), ms)
        client = TestClient(app)
        client.get("/messages/peer")
        ms.get_history.assert_called_once_with("peer", limit=100)

    def test_multiple_messages_ordered(self) -> None:
        msgs = [
            _make_stored_message(msg_id=i, plaintext=f"msg{i}".encode())
            for i in range(1, 4)
        ]
        client = _build_client(history=msgs)
        resp = client.get("/messages/any")
        ids = [m["id"] for m in resp.json()]
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_allows_requests_under_limit(self) -> None:
        client = _build_client(requests_per_minute=5)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_blocks_at_rate_limit(self) -> None:
        client = _build_client(requests_per_minute=5)
        for _ in range(5):
            client.get("/health")
        resp = client.get("/health")
        assert resp.status_code == 429

    def test_rate_limit_detail_message(self) -> None:
        client = _build_client(requests_per_minute=1)
        client.get("/health")
        resp = client.get("/health")
        assert resp.json()["detail"] == "rate limit exceeded"

    def test_rate_limit_zero_disables_limiting(self) -> None:
        # requests_per_minute=0 disables rate limiting entirely
        client = _build_client(requests_per_minute=0)
        for _ in range(200):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_rate_limit_resets_after_window(self) -> None:
        """Each TestClient has its own fresh bucket — new client is fresh."""
        client1 = _build_client(requests_per_minute=2)
        client1.get("/health")
        client1.get("/health")
        assert client1.get("/health").status_code == 429

        # Brand-new client = brand-new app = empty bucket
        client2 = _build_client(requests_per_minute=2)
        assert client2.get("/health").status_code == 200

    def test_rate_limit_applies_to_all_endpoints(self) -> None:
        client = _build_client(requests_per_minute=2)
        client.get("/health")
        client.get("/peers")
        resp = client.get("/routes")
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Message size cap
# ---------------------------------------------------------------------------


class TestMessageSizeCap:
    def test_message_within_limit_accepted(self) -> None:
        client = _build_client()
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "x" * 100},
        )
        assert resp.status_code == 200

    def test_message_at_limit_accepted(self) -> None:
        # 4096 ASCII chars = 4096 bytes
        client = _build_client()
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "x" * 4096},
        )
        assert resp.status_code == 200

    def test_message_over_limit_returns_413(self) -> None:
        client = _build_client()
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "x" * 4097},
        )
        assert resp.status_code == 413

    def test_413_detail_mentions_too_large(self) -> None:
        client = _build_client()
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "x" * 5000},
        )
        assert "too large" in resp.json()["detail"]

    def test_413_detail_mentions_byte_count(self) -> None:
        client = _build_client()
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": "x" * 5000},
        )
        assert "5000" in resp.json()["detail"]

    def test_empty_message_accepted(self) -> None:
        client = _build_client()
        resp = client.post(
            "/send",
            json={"peer_id": "ff00" * 8, "message": ""},
        )
        assert resp.status_code == 200
