"""
mesh/discovery.py — MESIM mDNS peer discovery

Advertises this node on the local network via mDNS and discovers other MESIM
nodes on the same WiFi/LAN. No sensitive data is broadcast — the mDNS record
contains only a partial device_id, rank, and protocol version. The full
PublicBundle (public keys) is exchanged during the transport handshake.

Service type: _mesim._udp.local.
TXT record:
  id   = first 16 hex chars of device_id (no hyphens)
  rank = str(int(rank))
  ver  = "1"
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Callable

from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

from core.identity import DeviceIdentity, PublicBundle, Rank

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_mesim._udp.local."
PROTOCOL_VER = b"1"


# ---------------------------------------------------------------------------
# PeerInfo
# ---------------------------------------------------------------------------


@dataclass
class PeerInfo:
    """
    Represents a discovered peer node.

    device_id is the 16-hex-char prefix from the mDNS TXT record.
    api_port is the REST API port advertised in the TXT record; used to
    fetch the PublicBundle before initiating a transport handshake.
    The full PublicBundle is populated by transport.py after the handshake.
    """

    device_id: str
    callsign: str | None
    rank: Rank
    addr: str
    port: int
    last_seen: float
    public_bundle: PublicBundle | None = None
    api_port: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_txt_properties(
    props: dict[bytes, bytes],
) -> tuple[str, Rank] | None:
    """
    Validate and extract identity fields from mDNS TXT properties.

    Returns (device_id_prefix: str, rank: Rank) or None if any field is
    missing, wrong version, or unparseable. Never raises.
    """
    try:
        if props.get(b"ver") != PROTOCOL_VER:
            return None
        id_raw = props.get(b"id")
        rank_raw = props.get(b"rank")
        if id_raw is None or rank_raw is None:
            return None
        device_id_prefix = id_raw.decode("ascii")
        rank_int = int(rank_raw.decode("ascii"))
        rank = Rank(rank_int)  # raises ValueError if out of 1-4 range
        return device_id_prefix, rank
    except (ValueError, KeyError, UnicodeDecodeError):
        return None


def _get_local_ip() -> str:
    """
    Determine the local non-loopback IP address by probing a UDP socket.
    Falls back to 127.0.0.1 if no suitable interface is found.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# ServiceListener
# ---------------------------------------------------------------------------


class _MeshServiceListener:
    """
    Implements the zeroconf ServiceListener protocol.
    Delegates to MeshDiscovery async handlers via asyncio.create_task.
    """

    def __init__(self, discovery: MeshDiscovery) -> None:
        self._discovery = discovery

    def add_service(self, zc, type_: str, name: str) -> None:
        asyncio.create_task(self._discovery._on_service_added(zc, type_, name))

    def remove_service(self, zc, type_: str, name: str) -> None:
        asyncio.create_task(self._discovery._on_service_removed(zc, type_, name))

    def update_service(self, zc, type_: str, name: str) -> None:
        asyncio.create_task(self._discovery._on_service_added(zc, type_, name))


# ---------------------------------------------------------------------------
# MeshDiscovery
# ---------------------------------------------------------------------------


class MeshDiscovery:
    """
    mDNS-based peer discovery for the MESIM mesh network.

    Usage::

        async with MeshDiscovery(identity) as disc:
            await disc.start(port=7777)
            disc.on_peer_discovered(lambda p: print("found:", p.callsign))
            # ... run until shutdown ...
    """

    def __init__(self, identity: DeviceIdentity) -> None:
        self._identity = identity
        self._own_id_prefix = identity.device_id.replace("-", "")[:16]
        self._peers: dict[str, PeerInfo] = {}
        self._svc_name_to_id: dict[str, str] = {}  # service name → device_id_prefix
        self._discovered_cbs: list[Callable[[PeerInfo], None]] = []
        self._lost_cbs: list[Callable[[str], None]] = []
        self._zeroconf: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._service_info: ServiceInfo | None = None
        self._lock = asyncio.Lock()

    # ── Context manager ─────────────────────────────────────────────────────

    async def __aenter__(self) -> MeshDiscovery:
        return self

    async def __aexit__(self, *_) -> None:
        if self._zeroconf is not None:
            await self.stop()

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self, port: int, api_port: int = 0) -> None:
        """
        Register this node on mDNS and start browsing for peers.

        Args:
            port:     UDP port this node is listening on.
            api_port: REST API port to advertise in the TXT record so peers
                      can fetch our PublicBundle before connecting.
        """
        local_ip = _get_local_ip()
        logger.debug(
            "mDNS starting on interface %s  UDP port=%d  API port=%d",
            local_ip, port, api_port,
        )
        # Bind zeroconf explicitly to the detected non-loopback interface so
        # mDNS packets reach other nodes on the same LAN/VLAN.  On a VPS the
        # default (loopback) prevents cross-node discovery entirely.
        self._zeroconf = AsyncZeroconf(interfaces=[local_ip])
        self._service_info = self._build_service_info(port, api_port)
        await self._zeroconf.async_register_service(self._service_info)
        logger.debug("mDNS advertisement registered: %s", self._service_info.name)

        listener = _MeshServiceListener(self)
        self._browser = AsyncServiceBrowser(
            self._zeroconf.zeroconf, SERVICE_TYPE, listener
        )
        logger.debug("mDNS peer browsing started for %s", SERVICE_TYPE)

    async def stop(self) -> None:
        """Unregister mDNS service and shut down zeroconf."""
        if self._zeroconf is None:
            return
        if self._service_info is not None:
            await self._zeroconf.async_unregister_service(self._service_info)
        await self._zeroconf.async_close()
        self._zeroconf = None
        self._browser = None

    # ── Public API ──────────────────────────────────────────────────────────

    def get_peers(self) -> dict[str, PeerInfo]:
        """Return a snapshot copy of the current peer cache."""
        return dict(self._peers)

    def on_peer_discovered(self, cb: Callable[[PeerInfo], None]) -> None:
        """Register a callback invoked when a new peer is found."""
        self._discovered_cbs.append(cb)

    def on_peer_lost(self, cb: Callable[[str], None]) -> None:
        """Register a callback invoked when a peer disappears (arg: device_id)."""
        self._lost_cbs.append(cb)

    def update_peer_bundle(self, device_id: str, bundle: PublicBundle) -> None:
        """
        Attach a verified PublicBundle to a known peer.
        Called by transport.py after a successful handshake.
        """
        if device_id in self._peers:
            self._peers[device_id].public_bundle = bundle

    # ── mDNS event handlers ─────────────────────────────────────────────────

    async def _on_service_added(self, zc, type_: str, name: str) -> None:
        """Handle a newly advertised or updated mDNS service."""
        info = ServiceInfo(type_, name)
        if not await info.async_request(zc, 3000):
            logger.debug("mDNS service_info request timed out for %s — skipping", name)
            return  # peer not responding

        parsed = _parse_txt_properties(info.properties or {})
        if parsed is None:
            logger.debug("mDNS TXT parse failed for %s — skipping", name)
            return  # not a valid MESIM service

        device_id_prefix, rank = parsed

        # Ignore our own advertisement
        if device_id_prefix == self._own_id_prefix:
            return

        addrs = info.parsed_addresses()
        if not addrs:
            logger.debug("mDNS no addresses for %s — skipping", name)
            return
        addr = addrs[0]

        # Parse optional api_port from TXT record
        props = info.properties or {}
        api_port_raw = props.get(b"api")
        try:
            api_port = int(api_port_raw.decode("ascii")) if api_port_raw else 0
        except (ValueError, UnicodeDecodeError):
            logger.debug("mDNS invalid api_port in TXT for %s — defaulting to 0", name)
            api_port = 0

        async with self._lock:
            is_new = device_id_prefix not in self._peers
            self._peers[device_id_prefix] = PeerInfo(
                device_id=device_id_prefix,
                callsign=None,
                rank=rank,
                addr=addr,
                port=info.port,
                last_seen=time.time(),
                public_bundle=self._peers.get(device_id_prefix, PeerInfo(
                    device_id_prefix, None, rank, addr, info.port, 0.0
                )).public_bundle,
                api_port=api_port,
            )
            self._svc_name_to_id[name] = device_id_prefix

        if is_new:
            peer = self._peers[device_id_prefix]
            logger.debug(
                "Peer discovered: %s at %s:%d (api_port=%d)",
                device_id_prefix, addr, info.port, api_port,
            )
            for cb in self._discovered_cbs:
                cb(peer)

    async def _on_service_removed(self, zc, type_: str, name: str) -> None:
        """Handle a peer going offline."""
        device_id_prefix = self._svc_name_to_id.pop(name, None)
        if device_id_prefix is None:
            return

        async with self._lock:
            self._peers.pop(device_id_prefix, None)

        for cb in self._lost_cbs:
            cb(device_id_prefix)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_service_info(self, port: int, api_port: int = 0) -> ServiceInfo:
        """Build the ServiceInfo object for mDNS registration."""
        host_ip = _get_local_ip()
        callsign = self._identity.callsign
        properties = {
            b"id":   self._own_id_prefix.encode("ascii"),
            b"rank": str(int(self._identity.rank)).encode("ascii"),
            b"ver":  PROTOCOL_VER,
            b"api":  str(api_port).encode("ascii"),
        }
        return ServiceInfo(
            type_=SERVICE_TYPE,
            name=f"{callsign}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(host_ip)],
            port=port,
            properties=properties,
            server=f"{callsign.lower()}.local.",
        )
