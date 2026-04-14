"""
api/server.py — MESIM FastAPI device server

Device-to-device REST API for local control and inter-device communication.
Intended to run on each field device; never exposed to the internet.

Endpoints:
  GET  /health             — liveness + identity info
  GET  /bundle             — our PublicBundle (base64) for peer handshake initiation
  GET  /peers              — active encrypted sessions
  GET  /routes             — BATMAN route table
  POST /send               — send or queue a message
  GET  /messages/{peer_id} — paginated message history

Security:
  - Rate limiting: max requests_per_minute (default 100) per client IP
  - Message size cap: MAX_MESSAGE_BYTES (4096) per /send payload
"""

from __future__ import annotations

import base64
import collections
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.identity import get_public_bundle

if TYPE_CHECKING:
    from core.identity import DeviceIdentity
    from core.store import MessageStore
    from mesh.router import MeshRouter
    from mesh.store_forward import StoreForward
    from mesh.transport import MeshTransport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_MESSAGE_BYTES = 4096  # maximum encoded payload per /send request


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SendRequest(BaseModel):
    peer_id: str = Field(..., description="Target device_id (hex, no hyphens)")
    message: str = Field(..., description="Plaintext message body (UTF-8)")


class SendResponse(BaseModel):
    peer_id: str
    queued: bool  # True = peer offline, message stored for later delivery
    msg_id: int   # row id in local MessageStore


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(
    identity: "DeviceIdentity",
    transport: "MeshTransport",
    router: "MeshRouter",
    store_forward: "StoreForward",
    message_store: "MessageStore",
    requests_per_minute: int = 100,
) -> FastAPI:
    """
    Build and return the FastAPI application.

    All subsystems are injected via app.state so endpoints remain fully
    testable without module-level singletons.

    Args:
        requests_per_minute: sliding-window rate limit per client IP.
                             Set to 0 to disable rate limiting.
    """
    app = FastAPI(
        title="MESIM Device API",
        version="1.0.0",
        description=(
            "Defense-grade encrypted P2P mesh — device-local control API. "
            "Bind to loopback or a trusted interface only."
        ),
    )
    app.state.identity = identity
    app.state.transport = transport
    app.state.router = router
    app.state.store_forward = store_forward
    app.state.message_store = message_store
    app.state.rate_buckets: dict[str, collections.deque] = {}

    # ── Rate-limiting middleware ────────────────────────────────────────────

    if requests_per_minute > 0:
        @app.middleware("http")
        async def _rate_limit(request: Request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            buckets = request.app.state.rate_buckets
            bucket = buckets.setdefault(client_ip, collections.deque())
            # evict timestamps older than 60 s
            while bucket and bucket[0] < now - 60:
                bucket.popleft()
            if len(bucket) >= requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                )
            bucket.append(now)
            return await call_next(request)

    # ── /health ────────────────────────────────────────────────────────────

    @app.get("/health")
    def health(request: Request) -> dict:
        """Liveness probe + local identity summary."""
        ident = request.app.state.identity
        return {
            "status": "ok",
            "device_id": ident.device_id,
            "callsign": ident.callsign,
            "rank": int(ident.rank),
            "rank_name": ident.rank.name,
            "timestamp": time.time(),
        }

    # ── /bundle ────────────────────────────────────────────────────────────

    @app.get("/bundle")
    def get_bundle(request: Request) -> dict:
        """
        Return this node's PublicBundle as JSON so peers can initiate a
        transport handshake without a prior mDNS TXT key exchange.

        All values are base64-encoded; the bundle_sig covers every other
        field and must be verified by the caller before use.
        """
        bundle = get_public_bundle(request.app.state.identity)
        return {
            "device_id":  bundle.device_id,
            "callsign":   bundle.callsign,
            "rank":       int(bundle.rank),
            "verify_key": base64.b64encode(bundle.verify_key.raw).decode(),
            "encrypt_pub": base64.b64encode(bundle.encrypt_pub.raw).decode(),
            "kem_pub":    base64.b64encode(bundle.kem_pub.raw).decode(),
            "bundle_sig": base64.b64encode(bundle.bundle_sig).decode(),
        }

    # ── /peers ─────────────────────────────────────────────────────────────

    @app.get("/peers")
    def list_peers(request: Request) -> list:
        """
        Return all active encrypted peer sessions.

        Each entry includes the peer's callsign, rank, address, session
        statistics, and when the session was established.
        """
        sessions = request.app.state.transport.get_sessions()
        return [
            {
                "peer_id": session.peer_id,
                "callsign": session.peer_bundle.callsign,
                "rank": int(session.peer_bundle.rank),
                "rank_name": session.peer_bundle.rank.name,
                "addr_host": session.peer_addr[0],
                "addr_port": session.peer_addr[1],
                "established_at": session.established_at,
                "last_active": session.last_active,
                "bytes_sent": session.bytes_sent,
                "bytes_recv": session.bytes_recv,
            }
            for session in sessions.values()
        ]

    # ── /routes ────────────────────────────────────────────────────────────

    @app.get("/routes")
    def list_routes(request: Request) -> list:
        """
        Return the BATMAN route table.

        Each entry shows the reachable destination, which direct neighbour to
        route through, hop distance, and the last originator sequence number.
        """
        table = request.app.state.router.get_route_table()
        return [
            {
                "originator_id": entry.originator_id,
                "next_hop_id": entry.next_hop_id,
                "next_hop_host": entry.next_hop_addr[0],
                "next_hop_port": entry.next_hop_addr[1],
                "hop_count": entry.hop_count,
                "last_seen": entry.last_seen,
                "seq_num": entry.seq_num,
            }
            for entry in table.values()
        ]

    # ── /send ──────────────────────────────────────────────────────────────

    @app.post("/send", response_model=SendResponse)
    async def send_message(body: SendRequest, request: Request) -> SendResponse:
        """
        Send a message to a peer.

        If the peer has an active session the message is delivered immediately
        (queued=False).  If the peer is offline the message is persisted in
        the store-and-forward queue and delivered automatically on reconnect
        (queued=True).

        The message is also saved to the local MessageStore in both cases so
        GET /messages/{peer_id} returns sent history.

        Returns 413 if the message exceeds MAX_MESSAGE_BYTES.
        Returns 503 if the underlying transport raises (e.g. queue full).
        """
        payload = body.message.encode("utf-8")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"message too large: {len(payload)} bytes (max {MAX_MESSAGE_BYTES})",
            )
        sf = request.app.state.store_forward
        ms = request.app.state.message_store
        try:
            delivered = await sf.send_or_queue(body.peer_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        msg_id = ms.save_message(body.peer_id, "sent", payload)
        return SendResponse(peer_id=body.peer_id, queued=not delivered, msg_id=msg_id)

    # ── /messages/{peer_id} ────────────────────────────────────────────────

    @app.get("/messages/{peer_id}")
    def get_messages(peer_id: str, request: Request, limit: int = 100) -> list:
        """
        Return decrypted message history for a peer, oldest first.

        plaintext is UTF-8 decoded when possible; binary payloads are
        base64-encoded so the JSON response is always valid.

        limit: max rows to return (default 100, capped at 500).
        """
        limit = min(limit, 500)
        ms = request.app.state.message_store
        history = ms.get_history(peer_id, limit=limit)
        result = []
        for msg in history:
            try:
                text = msg.plaintext.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = base64.b64encode(msg.plaintext).decode("ascii")
                encoding = "base64"
            result.append(
                {
                    "id": msg.id,
                    "direction": msg.direction,
                    "plaintext": text,
                    "encoding": encoding,
                    "timestamp": msg.timestamp,
                    "delivered": msg.delivered,
                }
            )
        return result

    return app
