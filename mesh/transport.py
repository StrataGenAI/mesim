"""
mesh/transport.py — MESIM encrypted UDP transport

Handles:
  - Packet framing (version + type + sender_id + payload_len + payload + MAC)
  - 3-step hybrid KEM handshake (HANDSHAKE → HANDSHAKE_RESP → KEY_INIT → KEY_ACK)
  - Authenticated encryption for all post-handshake packets
  - Message fragmentation and reassembly (always 6-byte frag header in MESSAGE)
  - Retry with exponential backoff (3 attempts: 0.5s, 1.0s, 2.0s)
  - ACK-based in-flight tracking

Packet wire format:
  [version:1B][type:1B][sender_id:32B][payload_len:2B][payload:NB][mac:16B]
  Total overhead: 52 bytes

MAC policy:
  HANDSHAKE / HANDSHAKE_RESP / KEY_INIT / KEY_ACK: ZERO_MAC (16 zero bytes)
    → authentication is via PublicBundle.bundle_sig (Ed25519)
  MESSAGE / ACK / PING / PONG / STORE_FORWARD: HMAC-SHA256[:16]
    → covers header + payload in constant-time comparison

MESSAGE payload layout (sign-then-encrypt, sig hidden inside ciphertext):
  [msg_id:4B][frag_idx:1B][frag_total:1B][nonce:12B]
  [ChaCha20-Poly1305( sign(plaintext, sk):64B + plaintext, session_key, aad=header )]

Always-present frag header: frag_total=1, frag_idx=0 for single-fragment messages.
This eliminates framing ambiguity at a cost of 6 bytes per packet.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_lib
import itertools
import os
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Awaitable, Callable

from cryptography.exceptions import InvalidSignature, InvalidTag

from core.crypto import (
    NonceCounter,
    SessionKey,
    decrypt_message,
    encrypt_message,
    kem_decapsulate,
    kem_encapsulate,
    sign_message,
    verify_signature,
)
from core.identity import DeviceIdentity, PublicBundle, Rank, verify_public_bundle

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 1
HEADER_FMT = "!BB32sH"          # version(1) + type(1) + sender_id(32) + payload_len(2)
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 36
MAC_SIZE = 16
OVERHEAD = HEADER_SIZE + MAC_SIZE           # 52
MTU = 1400
MAX_PAYLOAD = MTU - OVERHEAD                # 1348
BUNDLE_WIRE_LEN = 1381                      # 36+32+32+1184+32+1+64 (device_id included)
FRAG_HEADER_SIZE = 6                        # msg_id(4)+frag_idx(1)+frag_total(1)
NONCE_SIZE = 12
TAG_SIZE = 16
SIG_SIZE = 64

# Max plaintext per fragment:
# MAX_PAYLOAD - FRAG_HEADER_SIZE - NONCE_SIZE - TAG_SIZE = 1348-6-12-16 = 1314
# (The 64-byte signature is encrypted *inside* the ciphertext, so it counts
#  against plaintext space too:  1314 - 64 = 1250 bytes of user plaintext.)
MAX_FRAG_PT = MAX_PAYLOAD - FRAG_HEADER_SIZE - NONCE_SIZE - TAG_SIZE - SIG_SIZE  # 1250

ZERO_MAC = b"\x00" * MAC_SIZE
KEY_ACK_MAGIC = b"MESIM_ACK"           # 9 bytes
RETRY_BACKOFFS = (0.5, 1.0, 2.0)
MAX_RETRIES = 3
HANDSHAKE_TIMEOUT = 10.0
RETRY_INTERVAL = 0.1                   # seconds between retry-loop sweeps
MAX_FRAG_TOTAL = 255                   # max fragments per message (1 byte field)
MAX_REASSEMBLY_BUFS = 64               # prevent buffer exhaustion


# ---------------------------------------------------------------------------
# PacketType
# ---------------------------------------------------------------------------


class PacketType(IntEnum):
    HANDSHAKE = 0x01
    HANDSHAKE_RESP = 0x02
    KEY_INIT = 0x03
    KEY_ACK = 0x04
    MESSAGE = 0x05
    ACK = 0x06
    PING = 0x07
    PONG = 0x08
    STORE_FORWARD = 0x09
    ORIGINATOR    = 0x0A


_HANDSHAKE_TYPES = frozenset({
    PacketType.HANDSHAKE,
    PacketType.HANDSHAKE_RESP,
    PacketType.KEY_INIT,
    PacketType.KEY_ACK,
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Active encrypted session with a peer."""

    peer_id: str               # device_id UUID4 (36 chars, with hyphens) — canonical form
    peer_addr: tuple[str, int]
    session_key: SessionKey
    nonce_counter: NonceCounter
    peer_bundle: PublicBundle
    established_at: float
    last_active: float
    bytes_sent: int = 0
    bytes_recv: int = 0


@dataclass
class _PendingHandshake:
    """State held during an in-progress handshake."""

    peer_addr: tuple[str, int]
    peer_bundle: PublicBundle | None = None
    our_session_key: SessionKey | None = None
    resp_event: asyncio.Event = field(default_factory=asyncio.Event)
    ack_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _InFlight:
    """A sent packet awaiting ACK."""

    msg_id: int
    packet_bytes: bytes          # full serialized packet (for retransmit)
    peer_addr: tuple[str, int]
    attempt: int = 0             # number of send attempts so far (0 = not yet sent by retry loop)
    next_retry_at: float = 0.0   # monotonic time of next retry


# ---------------------------------------------------------------------------
# Wire-format helpers
# ---------------------------------------------------------------------------


def sender_id_bytes(device_id: str) -> bytes:
    """Encode device_id as 32 ASCII hex bytes (no hyphens)."""
    return device_id.replace("-", "")[:32].encode("ascii").ljust(32, b"0")


def device_id_from_bytes(raw: bytes) -> str:
    """Decode 32-byte sender_id field back to a hex string."""
    return raw.decode("ascii")


def pack_bundle(bundle: PublicBundle) -> bytes:
    """
    Serialize a PublicBundle to 1381 bytes for wire transmission.

    Layout:
      device_id(36, zero-padded UTF-8) + verify_key(32) + encrypt_pub(32)
      + kem_pub(1184) + callsign(32, zero-padded) + rank(1) + bundle_sig(64)

    device_id is included because bundle_sig covers it (via _canonical_bundle_bytes).
    """
    device_id_b = bundle.device_id.encode("utf-8").ljust(36, b"\x00")[:36]
    callsign_b = bundle.callsign.encode("ascii").ljust(32, b"\x00")[:32]
    return (
        device_id_b
        + bundle.verify_key.raw
        + bundle.encrypt_pub.raw
        + bundle.kem_pub.raw
        + callsign_b
        + struct.pack("!B", int(bundle.rank))
        + bundle.bundle_sig
    )


def unpack_bundle(data: bytes) -> PublicBundle:
    """
    Deserialize 1381 bytes to a PublicBundle.

    Raises ValueError on wrong length.
    Caller must call verify_public_bundle() before trusting the result.
    """
    if len(data) != BUNDLE_WIRE_LEN:
        raise ValueError(f"Bundle wire data must be {BUNDLE_WIRE_LEN} bytes, got {len(data)}")

    from core.crypto import Ed25519VerifyKey, MLKEMPublicKey, X25519PubKey

    offset = 0
    device_id = data[offset: offset + 36].rstrip(b"\x00").decode("utf-8"); offset += 36
    verify_key = Ed25519VerifyKey(raw=data[offset: offset + 32]); offset += 32
    encrypt_pub = X25519PubKey(raw=data[offset: offset + 32]); offset += 32
    kem_pub = MLKEMPublicKey(raw=data[offset: offset + 1184]); offset += 1184
    callsign = data[offset: offset + 32].rstrip(b"\x00").decode("ascii"); offset += 32
    rank = Rank(struct.unpack_from("!B", data, offset)[0]); offset += 1
    bundle_sig = data[offset: offset + 64]

    return PublicBundle(
        device_id=device_id,
        callsign=callsign,
        rank=rank,
        verify_key=verify_key,
        encrypt_pub=encrypt_pub,
        kem_pub=kem_pub,
        bundle_sig=bundle_sig,
    )


def compute_mac(session_key: SessionKey, header: bytes, payload: bytes) -> bytes:
    """HMAC-SHA256[:16] over header + payload using session_key."""
    return hmac_lib.new(
        session_key.raw,
        header + payload,
        hashlib.sha256,
    ).digest()[:MAC_SIZE]


def verify_mac(session_key: SessionKey, packet: bytes) -> None:
    """
    Verify the MAC on a packet.

    Raises InvalidTag on mismatch. Uses constant-time comparison.
    Must be called before decryption to prevent chosen-ciphertext attacks.
    """
    header = packet[:HEADER_SIZE]
    payload = packet[HEADER_SIZE: -MAC_SIZE]
    embedded_mac = packet[-MAC_SIZE:]
    expected = compute_mac(session_key, header, payload)
    if not hmac_lib.compare_digest(expected, embedded_mac):
        raise InvalidTag()


def build_packet(
    ptype: PacketType,
    sender_id: bytes,          # 32 bytes
    payload: bytes,
    session_key: SessionKey | None = None,
) -> bytes:
    """
    Assemble a complete wire packet.

    session_key=None → ZERO_MAC (used for handshake packets).
    """
    header = struct.pack(HEADER_FMT, PROTOCOL_VERSION, int(ptype), sender_id, len(payload))
    if session_key is None or ptype in _HANDSHAKE_TYPES:
        mac = ZERO_MAC
    else:
        mac = compute_mac(session_key, header, payload)
    return header + payload + mac


def parse_packet(data: bytes) -> tuple[int, PacketType, bytes, bytes]:
    """
    Parse raw bytes into (version, ptype, sender_id_bytes, payload).

    Raises ValueError on undersized data or unsupported protocol version.
    Does NOT verify the MAC — caller is responsible.
    """
    if len(data) < OVERHEAD:
        raise ValueError(f"Packet too short: {len(data)} < {OVERHEAD}")
    version, ptype_raw, sender_id, payload_len = struct.unpack_from(HEADER_FMT, data)
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {version}")
    payload = data[HEADER_SIZE: HEADER_SIZE + payload_len]
    return version, PacketType(ptype_raw), sender_id, payload


# ---------------------------------------------------------------------------
# _UDPProtocol — asyncio DatagramProtocol
# ---------------------------------------------------------------------------


class _UDPProtocol(asyncio.DatagramProtocol):
    """Thin asyncio DatagramProtocol that delegates to MeshTransport."""

    def __init__(self, transport_obj: MeshTransport) -> None:
        self._t = transport_obj

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        asyncio.create_task(self._t._dispatch(data, addr))

    def error_received(self, exc: Exception) -> None:
        pass  # log in production; ignore in Phase 1

    def connection_lost(self, exc: Exception | None) -> None:
        pass


# ---------------------------------------------------------------------------
# MeshTransport
# ---------------------------------------------------------------------------


class MeshTransport:
    """
    Encrypted UDP transport for MESIM mesh nodes.

    Usage::

        t = MeshTransport(identity)
        await t.start("0.0.0.0", 7777)
        t.on_message(my_handler)
        success = await t.connect(("192.168.1.5", 7777), peer_bundle)
        if success:
            await t.send(peer_device_id, b"hello")
        await t.stop()
    """

    def __init__(self, identity: DeviceIdentity) -> None:
        self._identity = identity
        self._sender_id = sender_id_bytes(identity.device_id)
        self._sessions: dict[str, SessionInfo] = {}          # peer_id → SessionInfo
        self._sessions_by_addr: dict[tuple, str] = {}        # addr → peer_id
        self._pending: dict[tuple, _PendingHandshake] = {}   # addr → handshake state
        self._in_flight: dict[int, _InFlight] = {}           # msg_id → _InFlight
        self._frag_bufs: dict[tuple, dict[int, bytes]] = {}
        # keyed by (sender_id_str, msg_id): {frag_idx: plaintext_fragment}
        self._msg_id_counter = itertools.count(1)
        self._udp_transport: asyncio.DatagramTransport | None = None
        self._message_cbs: list[Callable] = []
        self._connected_cbs: list[Callable] = []
        self._disconnected_cbs: list[Callable] = []
        self._send_failed_cbs: list[Callable] = []
        self._ack_cbs: list[Callable] = []
        self._raw_packet_cbs: dict[PacketType, Callable] = {}
        self._retry_task: asyncio.Task | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self, host: str, port: int) -> None:
        """Bind UDP socket and start the retry background task."""
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self),
            local_addr=(host, port),
        )
        self._udp_transport = transport
        self._retry_task = asyncio.create_task(self._retry_loop())

    async def stop(self) -> None:
        """Close the UDP socket and cancel background tasks."""
        if self._retry_task is not None:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
            self._retry_task = None
        if self._udp_transport is not None:
            self._udp_transport.close()
            self._udp_transport = None

    # ── Connect (initiator) ──────────────────────────────────────────────────

    async def connect(
        self, peer_addr: tuple[str, int], peer_bundle: PublicBundle
    ) -> bool:
        """
        Initiate a 3-step handshake with a peer.

        Verifies peer_bundle before sending anything.
        Returns True on success, False on timeout.
        Raises InvalidSignature if peer_bundle is tampered.
        """
        verify_public_bundle(peer_bundle)  # raises InvalidSignature if tampered

        pending = _PendingHandshake(peer_addr=peer_addr, peer_bundle=peer_bundle)
        self._pending[peer_addr] = pending

        # Step 1: send our bundle
        our_bundle = self._get_public_bundle()
        payload = pack_bundle(our_bundle)
        pkt = build_packet(PacketType.HANDSHAKE, self._sender_id, payload)
        self._send_raw(pkt, peer_addr)

        # Wait for HANDSHAKE_RESP (step 2)
        try:
            await asyncio.wait_for(pending.resp_event.wait(), timeout=HANDSHAKE_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(peer_addr, None)
            return False

        # Step 3: KEM encapsulation
        encap_result, session_key = kem_encapsulate(
            peer_bundle.encrypt_pub,
            peer_bundle.kem_pub,
            self._identity.encrypt_keypair.private_key,
        )
        pending.our_session_key = session_key

        kem_pkt = build_packet(PacketType.KEY_INIT, self._sender_id, encap_result.ciphertext)
        self._send_raw(kem_pkt, peer_addr)

        # Wait for KEY_ACK (step 4)
        try:
            await asyncio.wait_for(pending.ack_event.wait(), timeout=HANDSHAKE_TIMEOUT)
        except asyncio.TimeoutError:
            self._pending.pop(peer_addr, None)
            return False

        self._pending.pop(peer_addr, None)
        return True

    # ── Send ────────────────────────────────────────────────────────────────

    async def send(self, peer_id: str, message: bytes) -> int:
        """
        Encrypt and send a message to a connected peer.

        Fragments automatically if message exceeds MAX_FRAG_PT.
        Returns the msg_id for ACK tracking.
        Raises KeyError if peer_id has no active session.
        """
        session = self._sessions[peer_id]
        msg_id = self._next_msg_id()

        sig = sign_message(message, self._identity.signing_keypair.signing_key)
        plaintext_with_sig = sig + message

        # Split into fragments
        chunks = [
            plaintext_with_sig[i: i + MAX_FRAG_PT]
            for i in range(0, len(plaintext_with_sig), MAX_FRAG_PT)
        ]
        frag_total = len(chunks)

        for frag_idx, chunk in enumerate(chunks):
            frag_header = struct.pack("!IBB", msg_id, frag_idx, frag_total)
            # Pre-compute pkt_header before encryption so it can serve as AAD,
            # matching the aad=full_packet[:HEADER_SIZE] used in _handle_message.
            payload_len = FRAG_HEADER_SIZE + NONCE_SIZE + len(chunk) + TAG_SIZE
            pkt_header = struct.pack(
                HEADER_FMT,
                PROTOCOL_VERSION,
                int(PacketType.MESSAGE),
                self._sender_id,
                payload_len,
            )
            nonce, ct = encrypt_message(chunk, session.session_key, aad=pkt_header)
            payload = frag_header + nonce + ct

            mac = compute_mac(session.session_key, pkt_header, payload)
            pkt = pkt_header + payload + mac

            self._in_flight[msg_id] = _InFlight(
                msg_id=msg_id,
                packet_bytes=pkt,
                peer_addr=session.peer_addr,
                attempt=1,
                next_retry_at=time.monotonic() + RETRY_BACKOFFS[0],
            )
            self._send_raw(pkt, session.peer_addr)

        session.bytes_sent += len(message)
        session.last_active = time.time()
        return msg_id

    # ── Callbacks ───────────────────────────────────────────────────────────

    def on_message(self, cb: Callable[[str, bytes], Awaitable[None]]) -> None:
        self._message_cbs.append(cb)

    def on_peer_connected(self, cb: Callable[[str], Awaitable[None]]) -> None:
        self._connected_cbs.append(cb)

    def on_peer_disconnected(self, cb: Callable[[str], Awaitable[None]]) -> None:
        self._disconnected_cbs.append(cb)

    def on_send_failed(self, cb: Callable[[int, tuple], None]) -> None:
        self._send_failed_cbs.append(cb)

    def on_ack_received(self, cb: Callable[[int], Awaitable[None]]) -> None:
        """Register a callback fired when an ACK is received for msg_id."""
        self._ack_cbs.append(cb)

    def get_sessions(self) -> dict[str, SessionInfo]:
        """Return a shallow copy of active sessions."""
        return dict(self._sessions)

    def on_raw_packet(
        self,
        ptype: PacketType,
        cb: Callable[[bytes, tuple, bytes], Awaitable[None]],
    ) -> None:
        """
        Register a handler for a specific PacketType not handled internally.

        cb(payload, addr, sender_id_bytes) is awaited when a packet of that
        type arrives after MAC verification (for established sessions) or
        without MAC check (for types that arrive before session setup).
        """
        self._raw_packet_cbs[ptype] = cb

    # ── Packet dispatch ──────────────────────────────────────────────────────

    async def _dispatch(self, data: bytes, addr: tuple[str, int]) -> None:
        """Route an incoming packet to the appropriate handler."""
        try:
            _version, ptype, sender_id, payload = parse_packet(data)
        except (ValueError, KeyError):
            return  # malformed — drop silently

        session = self._sessions_by_addr.get(addr)

        # For established sessions, verify MAC before anything else
        if session and ptype not in _HANDSHAKE_TYPES:
            sess = self._sessions.get(session)
            if sess is None:
                return
            try:
                verify_mac(sess.session_key, data)
            except InvalidTag:
                return  # drop — never respond to failed auth

        try:
            if ptype == PacketType.HANDSHAKE:
                await self._handle_handshake(payload, addr, sender_id)
            elif ptype == PacketType.HANDSHAKE_RESP:
                await self._handle_handshake_resp(payload, addr, sender_id)
            elif ptype == PacketType.KEY_INIT:
                await self._handle_key_init(payload, addr, sender_id)
            elif ptype == PacketType.KEY_ACK:
                await self._handle_key_ack(payload, addr)
            elif ptype == PacketType.MESSAGE:
                await self._handle_message(payload, addr, sender_id, data)
            elif ptype == PacketType.ACK:
                await self._handle_ack(payload, addr)
            elif ptype == PacketType.PING:
                sess = self._sessions.get(self._sessions_by_addr.get(addr, ""))
                if sess:
                    await self._handle_ping(addr, sess)
            elif ptype == PacketType.PONG:
                pass  # update last_active in future
            elif ptype in self._raw_packet_cbs:
                cb = self._raw_packet_cbs[ptype]
                result = cb(payload, addr, sender_id)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
        except (InvalidSignature, InvalidTag, ValueError):
            return  # drop silently — never leak error info

    # ── Handshake handlers ───────────────────────────────────────────────────

    async def _handle_handshake(
        self, payload: bytes, addr: tuple, sender_id: bytes
    ) -> None:
        """Responder: received HANDSHAKE from initiator. Send HANDSHAKE_RESP."""
        if len(payload) != BUNDLE_WIRE_LEN:
            return
        initiator_bundle = unpack_bundle(payload)
        verify_public_bundle(initiator_bundle)  # drop if invalid

        # Store for KEY_INIT
        pending = self._pending.get(addr) or _PendingHandshake(peer_addr=addr)
        pending.peer_bundle = initiator_bundle
        self._pending[addr] = pending

        our_bundle = self._get_public_bundle()
        pkt = build_packet(PacketType.HANDSHAKE_RESP, self._sender_id, pack_bundle(our_bundle))
        self._send_raw(pkt, addr)

    async def _handle_handshake_resp(
        self, payload: bytes, addr: tuple, sender_id: bytes
    ) -> None:
        """Initiator: received HANDSHAKE_RESP. Signal resp_event."""
        pending = self._pending.get(addr)
        if pending is None:
            return
        if len(payload) != BUNDLE_WIRE_LEN:
            return
        resp_bundle = unpack_bundle(payload)
        verify_public_bundle(resp_bundle)

        # Update peer_bundle with the verified responder bundle (preserves
        # the one passed into connect() but fills in device_id)
        pending.peer_bundle = resp_bundle
        pending.resp_event.set()

    async def _handle_key_init(
        self, payload: bytes, addr: tuple, sender_id: bytes
    ) -> None:
        """Responder: received KEM ciphertext. Decapsulate → session key."""
        pending = self._pending.get(addr)
        if pending is None or pending.peer_bundle is None:
            return

        from core.crypto import _MLKEM_CT_LEN
        if len(payload) != _MLKEM_CT_LEN:
            return

        initiator_bundle = pending.peer_bundle
        session_key = kem_decapsulate(
            kem_ciphertext=payload,
            initiator_x25519_pub=initiator_bundle.encrypt_pub,
            our_x25519_priv=self._identity.encrypt_keypair.private_key,
            our_mlkem_secret=self._identity.kem_keypair.secret_key,
        )

        # Send KEY_ACK
        nonce, ct = encrypt_message(KEY_ACK_MAGIC + os.urandom(8), session_key)
        ack_pkt = build_packet(PacketType.KEY_ACK, self._sender_id, nonce + ct)
        self._send_raw(ack_pkt, addr)

        # Establish session — use bundle.device_id (UUID4 with hyphens) so both
        # sides of the handshake key their session by the same canonical form.
        peer_id = initiator_bundle.device_id
        self._sessions[peer_id] = SessionInfo(
            peer_id=peer_id,
            peer_addr=addr,
            session_key=session_key,
            nonce_counter=NonceCounter(),
            peer_bundle=initiator_bundle,
            established_at=time.time(),
            last_active=time.time(),
        )
        self._sessions_by_addr[addr] = peer_id
        self._pending.pop(addr, None)

        await self._fire_cbs(self._connected_cbs, peer_id)

    async def _handle_key_ack(self, payload: bytes, addr: tuple) -> None:
        """Initiator: received KEY_ACK. Verify and establish session."""
        pending = self._pending.get(addr)
        if pending is None or pending.our_session_key is None:
            return
        if len(payload) < NONCE_SIZE + len(KEY_ACK_MAGIC) + TAG_SIZE:
            return

        nonce = payload[:NONCE_SIZE]
        ct = payload[NONCE_SIZE:]
        try:
            pt = decrypt_message(nonce, ct, pending.our_session_key)
        except InvalidTag:
            return  # wrong key or tampered — drop

        if not pt.startswith(KEY_ACK_MAGIC):
            return

        if pending.peer_bundle is None:
            return
        peer_id = pending.peer_bundle.device_id

        self._sessions[peer_id] = SessionInfo(
            peer_id=peer_id,
            peer_addr=addr,
            session_key=pending.our_session_key,
            nonce_counter=NonceCounter(),
            peer_bundle=pending.peer_bundle,
            established_at=time.time(),
            last_active=time.time(),
        )
        self._sessions_by_addr[addr] = peer_id
        pending.ack_event.set()

        await self._fire_cbs(self._connected_cbs, peer_id)

    # ── Message handler ──────────────────────────────────────────────────────

    async def _handle_message(
        self,
        payload: bytes,
        addr: tuple,
        sender_id: bytes,
        full_packet: bytes,
    ) -> None:
        """
        Decrypt, verify signature, reassemble fragments, dispatch to callbacks.

        MAC already verified by _dispatch before this is called.
        """
        peer_id_str = self._sessions_by_addr.get(addr)
        if peer_id_str is None:
            return
        session = self._sessions.get(peer_id_str)
        if session is None:
            return

        # Parse frag header
        if len(payload) < FRAG_HEADER_SIZE + NONCE_SIZE + TAG_SIZE:
            return

        msg_id, frag_idx, frag_total = struct.unpack_from("!IBB", payload, 0)
        if frag_total == 0 or frag_idx >= frag_total:
            return  # invalid framing — drop

        nonce = payload[FRAG_HEADER_SIZE: FRAG_HEADER_SIZE + NONCE_SIZE]
        ct = payload[FRAG_HEADER_SIZE + NONCE_SIZE:]

        # AAD = packet header (first HEADER_SIZE bytes of full_packet)
        aad = full_packet[:HEADER_SIZE]
        try:
            fragment_with_sig = decrypt_message(nonce, ct, session.session_key, aad=aad)
        except InvalidTag:
            return  # tampered or replayed — drop silently

        # Store fragment
        buf_key = (peer_id_str, msg_id)
        if buf_key not in self._frag_bufs:
            if len(self._frag_bufs) >= MAX_REASSEMBLY_BUFS:
                return  # prevent buffer exhaustion
            self._frag_bufs[buf_key] = {}
        self._frag_bufs[buf_key][frag_idx] = fragment_with_sig

        # Check if all fragments arrived
        if len(self._frag_bufs[buf_key]) < frag_total:
            return  # waiting for more

        # Reassemble
        assembled = b"".join(
            self._frag_bufs[buf_key][i] for i in range(frag_total)
        )
        del self._frag_bufs[buf_key]

        # Extract signature (first SIG_SIZE bytes) and plaintext
        if len(assembled) < SIG_SIZE:
            return
        sig = assembled[:SIG_SIZE]
        plaintext = assembled[SIG_SIZE:]

        # Verify sender signature
        try:
            verify_signature(plaintext, sig, session.peer_bundle.verify_key)
        except InvalidSignature:
            return  # drop — forged or corrupted message

        # Send ACK
        ack_payload = struct.pack("!I", msg_id)
        ack_pkt = build_packet(
            PacketType.ACK, self._sender_id, ack_payload, session.session_key
        )
        self._send_raw(ack_pkt, addr)

        session.bytes_recv += len(plaintext)
        session.last_active = time.time()

        # peer_id_str is the canonical UUID4 key (with hyphens) from _sessions_by_addr
        await self._fire_cbs(self._message_cbs, peer_id_str, plaintext)

    async def _handle_ack(self, payload: bytes, addr: tuple) -> None:
        """Remove an acknowledged message from the in-flight tracker."""
        if len(payload) < 4:
            return
        msg_id = struct.unpack_from("!I", payload, 0)[0]
        self._in_flight.pop(msg_id, None)
        await self._fire_cbs(self._ack_cbs, msg_id)

    async def _handle_ping(self, addr: tuple, session: SessionInfo) -> None:
        """Respond to a PING with a PONG."""
        pkt = build_packet(PacketType.PONG, self._sender_id, b"", session.session_key)
        self._send_raw(pkt, addr)
        session.last_active = time.time()

    # ── Retry loop ───────────────────────────────────────────────────────────

    async def _retry_loop(self) -> None:
        """
        Background task: retransmit unACKed messages with exponential backoff.
        Fires on_send_failed after MAX_RETRIES exhausted.
        """
        while True:
            await asyncio.sleep(RETRY_INTERVAL)
            now = time.monotonic()
            expired = [
                inf for inf in list(self._in_flight.values())
                if now >= inf.next_retry_at
            ]
            for inf in expired:
                if inf.attempt >= MAX_RETRIES:
                    self._in_flight.pop(inf.msg_id, None)
                    for cb in self._send_failed_cbs:
                        cb(inf.msg_id, inf.peer_addr)
                else:
                    self._send_raw(inf.packet_bytes, inf.peer_addr)
                    inf.next_retry_at = now + RETRY_BACKOFFS[inf.attempt]
                    inf.attempt += 1

    # ── Private helpers ──────────────────────────────────────────────────────

    def _send_raw(self, packet: bytes, addr: tuple[str, int]) -> None:
        """Send a raw packet. No-op if transport not started."""
        if self._udp_transport is not None:
            self._udp_transport.sendto(packet, addr)

    def _next_msg_id(self) -> int:
        return next(self._msg_id_counter)

    def _get_public_bundle(self) -> PublicBundle:
        from core.identity import get_public_bundle
        return get_public_bundle(self._identity)

    def _get_session_for_addr(self, addr: tuple[str, int]) -> SessionInfo | None:
        peer_id = self._sessions_by_addr.get(addr)
        if peer_id is None:
            return None
        return self._sessions.get(peer_id)

    async def _fire_cbs(self, cbs: list, *args) -> None:
        """Dispatch callbacks. Coroutine callbacks are scheduled as tasks."""
        for cb in cbs:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _bundle_with_device_id(bundle: PublicBundle, sender_id: bytes) -> PublicBundle:
    """Return a new PublicBundle with device_id set from the sender_id field."""
    return PublicBundle(
        device_id=device_id_from_bytes(sender_id),
        callsign=bundle.callsign,
        rank=bundle.rank,
        verify_key=bundle.verify_key,
        encrypt_pub=bundle.encrypt_pub,
        kem_pub=bundle.kem_pub,
        bundle_sig=bundle.bundle_sig,
    )
