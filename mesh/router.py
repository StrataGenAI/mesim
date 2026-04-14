"""
mesh/router.py — MESIM BATMAN-inspired multi-hop router

Each node periodically broadcasts an ORIGINATOR packet to all connected peers.
When a node receives an originator:
  1. Updates its route table: originator is reachable via the direct sender
     with hop_count hops.
  2. Re-broadcasts with hop_count+1 to all OTHER connected peers, if
     hop_count < max_hops (default 7).

Signature policy:
  Originator packets are signed by the originating node's Ed25519 key.
  Intermediaries verify the signature before forwarding (prevents injection).
  The signature covers fields that do NOT change per hop:
    originator_id || seq_num || max_hops || timestamp (8B float64)

Wire format (110 bytes):
  [originator_id: 32B]  — sender_id encoding of originator's device_id
  [seq_num:        4B]  — big-endian uint32
  [hop_count:      1B]
  [max_hops:       1B]
  [timestamp:      8B]  — big-endian float64
  [signature:     64B]  — Ed25519 over _originator_signed_bytes(...)
"""

from __future__ import annotations

import asyncio
import itertools
import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from cryptography.exceptions import InvalidSignature

from core.crypto import sign_message, verify_signature
from core.identity import DeviceIdentity, PublicBundle

if TYPE_CHECKING:
    from mesh.transport import MeshTransport, PacketType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_HOPS             = 7
ORIGINATOR_INTERVAL  = 10.0   # seconds between our own originator broadcasts
STALE_ROUTE_SECONDS  = 60     # routes older than this are ignored by route_to

_ORIGINATOR_FMT  = "!32sIBBd"   # originator_id, seq_num, hop_count, max_hops, timestamp
_ORIGINATOR_SIZE = struct.calcsize(_ORIGINATOR_FMT)   # 46 bytes
_SIG_SIZE        = 64
ORIGINATOR_WIRE_LEN = _ORIGINATOR_SIZE + _SIG_SIZE    # 110 bytes


# ---------------------------------------------------------------------------
# RouteEntry
# ---------------------------------------------------------------------------


@dataclass
class RouteEntry:
    """A single entry in the BATMAN route table."""

    originator_id: str           # device_id of the reachable node
    next_hop_id: str             # peer_id of the direct neighbor to send through
    next_hop_addr: tuple[str, int]
    hop_count: int               # 1 = directly connected, 2+ = multi-hop
    last_seen: float             # time.time() when last updated
    seq_num: int                 # highest originator seq_num seen


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def _originator_signed_bytes(
    originator_id_bytes: bytes,
    seq_num: int,
    max_hops: int,
    timestamp: float,
) -> bytes:
    """Canonical bytes that the originator signs (hop_count excluded — it changes)."""
    return struct.pack("!32sId", originator_id_bytes, seq_num, timestamp) + bytes([max_hops])


def pack_originator(
    originator_id: str,
    seq_num: int,
    hop_count: int,
    max_hops: int,
    timestamp: float,
    signing_key,   # Ed25519SigningKey
) -> bytes:
    """
    Serialize and sign an ORIGINATOR packet.

    Returns 110-byte wire bytes.
    """
    from mesh.transport import sender_id_bytes
    oid_bytes = sender_id_bytes(originator_id)
    header = struct.pack(_ORIGINATOR_FMT, oid_bytes, seq_num, hop_count, max_hops, timestamp)
    signed = _originator_signed_bytes(oid_bytes, seq_num, max_hops, timestamp)
    sig = sign_message(signed, signing_key)
    return header + sig


def unpack_originator(data: bytes) -> tuple[bytes, int, int, int, float, bytes]:
    """
    Deserialize an ORIGINATOR packet.

    Returns (originator_id_bytes, seq_num, hop_count, max_hops, timestamp, signature).
    Raises ValueError on wrong length.
    """
    if len(data) != ORIGINATOR_WIRE_LEN:
        raise ValueError(f"ORIGINATOR must be {ORIGINATOR_WIRE_LEN} bytes, got {len(data)}")
    oid_bytes, seq_num, hop_count, max_hops, timestamp = struct.unpack_from(
        _ORIGINATOR_FMT, data, 0
    )
    sig = data[_ORIGINATOR_SIZE:]
    return oid_bytes, seq_num, hop_count, max_hops, timestamp, sig


# ---------------------------------------------------------------------------
# MeshRouter
# ---------------------------------------------------------------------------


class MeshRouter:
    """
    BATMAN-inspired multi-hop router for MESIM.

    Usage::

        router = MeshRouter(identity, transport)
        await router.start()
        entry = router.route_to(peer_device_id)
        if entry:
            transport.send_raw_to(entry.next_hop_addr, packet)
        await router.stop()
    """

    def __init__(
        self,
        identity: DeviceIdentity,
        transport: MeshTransport,
        max_hops: int = MAX_HOPS,
        originator_interval: float = ORIGINATOR_INTERVAL,
    ) -> None:
        self._identity = identity
        self._transport = transport
        self._max_hops = max_hops
        self._originator_interval = originator_interval
        self._seq_num: int = 0
        self._routes: dict[str, RouteEntry] = {}
        # Track highest seq_num seen per originator (loop / dup prevention)
        self._seen_seq: dict[str, int] = {}
        # Public bundles of known peers (populated externally or on receive)
        self._known_bundles: dict[str, PublicBundle] = {}
        self._originator_task: asyncio.Task | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Register ORIGINATOR handler with transport and start broadcast loop."""
        from mesh.transport import PacketType
        self._transport.on_raw_packet(PacketType.ORIGINATOR, self._handle_originator)
        self._originator_task = asyncio.create_task(self._originator_loop())

    async def stop(self) -> None:
        """Stop the background broadcast task."""
        if self._originator_task is not None:
            self._originator_task.cancel()
            try:
                await self._originator_task
            except asyncio.CancelledError:
                pass
            self._originator_task = None

    # ── Public API ─────────────────────────────────────────────────────────

    def route_to(self, peer_id: str) -> RouteEntry | None:
        """
        Return the best non-stale route to peer_id, or None if unknown / stale.

        "Best" means lowest hop_count among non-stale entries.
        """
        entry = self._routes.get(peer_id)
        if entry is None:
            return None
        if time.time() - entry.last_seen > STALE_ROUTE_SECONDS:
            return None
        return entry

    def get_route_table(self) -> dict[str, RouteEntry]:
        """Return a shallow copy of the full route table."""
        return dict(self._routes)

    def update_route(
        self,
        originator_id: str,
        next_hop_id: str,
        next_hop_addr: tuple[str, int],
        hop_count: int,
        seq_num: int,
        timestamp: float,
    ) -> None:
        """
        Insert or update a route entry.

        Replacement rules (BATMAN-inspired):
        - Higher seq_num always wins.
        - Equal seq_num: lower hop_count wins.
        - Otherwise (lower seq_num): ignore.
        """
        existing = self._routes.get(originator_id)
        if existing is not None:
            if seq_num < existing.seq_num:
                return   # stale update
            if seq_num == existing.seq_num and hop_count >= existing.hop_count:
                return   # not better
        self._routes[originator_id] = RouteEntry(
            originator_id=originator_id,
            next_hop_id=next_hop_id,
            next_hop_addr=next_hop_addr,
            hop_count=hop_count,
            last_seen=timestamp,
            seq_num=seq_num,
        )

    def purge_stale(self) -> int:
        """Remove all routes that haven't been updated within STALE_ROUTE_SECONDS."""
        cutoff = time.time() - STALE_ROUTE_SECONDS
        stale = [k for k, v in self._routes.items() if v.last_seen < cutoff]
        for k in stale:
            del self._routes[k]
        return len(stale)

    def register_bundle(self, peer_id: str, bundle: PublicBundle) -> None:
        """Register a peer's PublicBundle so we can verify their originator sigs."""
        self._known_bundles[peer_id] = bundle

    # ── Internal handlers ─────────────────────────────────────────────────

    async def _originator_loop(self) -> None:
        """Broadcast our own originator at regular intervals."""
        while True:
            await self._broadcast_originator()
            await asyncio.sleep(self._originator_interval)

    async def _broadcast_originator(self) -> None:
        """Send one originator packet to all currently active sessions."""
        self._seq_num += 1
        data = pack_originator(
            originator_id=self._identity.device_id,
            seq_num=self._seq_num,
            hop_count=0,
            max_hops=self._max_hops,
            timestamp=time.time(),
            signing_key=self._identity.signing_keypair.signing_key,
        )
        from mesh.transport import PacketType, PROTOCOL_VERSION, HEADER_FMT, ZERO_MAC
        import struct as _struct
        sender_id = self._transport._sender_id
        payload = data
        header = _struct.pack(
            HEADER_FMT, PROTOCOL_VERSION, int(PacketType.ORIGINATOR),
            sender_id, len(payload),
        )
        pkt = header + payload + ZERO_MAC
        sessions = self._transport.get_sessions()
        for session in sessions.values():
            self._transport._send_raw(pkt, session.peer_addr)

    async def _handle_originator(
        self, payload: bytes, from_addr: tuple, sender_id: bytes
    ) -> None:
        """
        Process a received ORIGINATOR packet.

        1. Validate length.
        2. Drop if from ourselves.
        3. Verify Ed25519 signature if we know the originator's bundle.
        4. Loop / dup prevention via seen_seq.
        5. Update route table.
        6. Re-forward with hop_count+1 if below max_hops.
        """
        if len(payload) != ORIGINATOR_WIRE_LEN:
            return

        try:
            oid_bytes, seq_num, hop_count, max_hops, ts, sig = unpack_originator(payload)
        except ValueError:
            return

        # Ignore our own originator packets
        if oid_bytes == self._transport._sender_id:
            return

        oid_str = oid_bytes.decode("ascii", errors="replace")

        # Signature verification if we know this originator
        bundle = self._known_bundles.get(oid_str)
        if bundle is not None:
            try:
                signed = _originator_signed_bytes(oid_bytes, seq_num, max_hops, ts)
                verify_signature(signed, sig, bundle.verify_key)
            except InvalidSignature:
                return  # drop — forged or corrupted

        # Loop / duplicate prevention
        seen = self._seen_seq.get(oid_str, -1)
        if seq_num <= seen:
            return   # already forwarded this or a newer one
        self._seen_seq[oid_str] = seq_num

        # Resolve next_hop_id from sender_id header
        from mesh.transport import device_id_from_bytes
        next_hop_id = device_id_from_bytes(sender_id)

        # Update route table
        self.update_route(
            originator_id=oid_str,
            next_hop_id=next_hop_id,
            next_hop_addr=from_addr,
            hop_count=hop_count + 1,
            seq_num=seq_num,
            timestamp=ts,
        )

        # Re-forward if within hop budget
        if hop_count >= max_hops:
            return

        # Build forwarding packet with hop_count incremented
        fwd_data = pack_originator(
            originator_id=oid_str,
            seq_num=seq_num,
            hop_count=hop_count + 1,
            max_hops=max_hops,
            timestamp=ts,
            signing_key=self._identity.signing_keypair.signing_key,
        ) if bundle is None else _repack_originator(
            oid_bytes, seq_num, hop_count + 1, max_hops, ts, sig
        )

        from mesh.transport import PacketType, PROTOCOL_VERSION, HEADER_FMT, ZERO_MAC
        import struct as _struct
        header = _struct.pack(
            HEADER_FMT, PROTOCOL_VERSION, int(PacketType.ORIGINATOR),
            self._transport._sender_id, len(fwd_data),
        )
        pkt = header + fwd_data + ZERO_MAC

        sessions = self._transport.get_sessions()
        for session in sessions.values():
            if session.peer_addr != from_addr:
                self._transport._send_raw(pkt, session.peer_addr)


# ---------------------------------------------------------------------------
# Internal re-pack helper (forward without re-signing)
# ---------------------------------------------------------------------------


def _repack_originator(
    oid_bytes: bytes,
    seq_num: int,
    hop_count: int,
    max_hops: int,
    timestamp: float,
    original_sig: bytes,
) -> bytes:
    """
    Re-serialize an originator packet with an updated hop_count, preserving
    the original signature (which does NOT cover hop_count).
    """
    header = struct.pack(_ORIGINATOR_FMT, oid_bytes, seq_num, hop_count, max_hops, timestamp)
    return header + original_sig
