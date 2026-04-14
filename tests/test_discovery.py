"""
tests/test_discovery.py — Unit tests for mesh/discovery.py

All zeroconf I/O is mocked. No real network required.
Run: pytest tests/test_discovery.py -v
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio

from core.identity import DeviceIdentity, PublicBundle, Rank, create_identity, get_public_bundle
from core.crypto import _oqs_available

requires_liboqs = pytest.mark.skipif(not _oqs_available, reason="liboqs not available")


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
async def bundle_b(identity_b):
    return get_public_bundle(identity_b)


def _make_valid_props(device_id: str, rank: Rank) -> dict:
    """Build valid mDNS TXT properties dict for a peer."""
    id_prefix = device_id.replace("-", "")[:16].encode()
    return {
        b"id": id_prefix,
        b"rank": str(int(rank)).encode(),
        b"ver": b"1",
    }


def _mock_service_info(device_id: str, rank: Rank, addr: str = "192.168.1.5", port: int = 7777):
    """Build a mock ServiceInfo object that returns valid data."""
    info = MagicMock()
    info.async_request = AsyncMock(return_value=True)
    info.parsed_addresses.return_value = [addr]
    info.port = port
    info.properties = _make_valid_props(device_id, rank)
    return info


# ---------------------------------------------------------------------------
# Lazy import after the module exists
# ---------------------------------------------------------------------------

def _import():
    from mesh.discovery import MeshDiscovery, PeerInfo, _parse_txt_properties
    return MeshDiscovery, PeerInfo, _parse_txt_properties


# ---------------------------------------------------------------------------
# Group 1 — PeerInfo dataclass
# ---------------------------------------------------------------------------


@requires_liboqs
def test_peer_info_creation_with_all_fields(bundle_b, identity_b):
    _, PeerInfo, _ = _import()
    now = time.time()
    peer = PeerInfo(
        device_id="abcd1234abcd1234",
        callsign="BRAVO-2",
        rank=Rank.SQUAD,
        addr="192.168.1.5",
        port=7777,
        last_seen=now,
        public_bundle=None,
    )
    assert peer.device_id == "abcd1234abcd1234"
    assert peer.callsign == "BRAVO-2"
    assert peer.rank == Rank.SQUAD
    assert peer.addr == "192.168.1.5"
    assert peer.port == 7777
    assert peer.last_seen == now
    assert peer.public_bundle is None


@requires_liboqs
def test_peer_info_public_bundle_defaults_none():
    _, PeerInfo, _ = _import()
    peer = PeerInfo(
        device_id="aaa",
        callsign=None,
        rank=Rank.NCO,
        addr="10.0.0.1",
        port=7777,
        last_seen=0.0,
    )
    assert peer.public_bundle is None


@requires_liboqs
def test_peer_info_last_seen_is_float():
    _, PeerInfo, _ = _import()
    peer = PeerInfo("x", None, Rank.SQUAD, "1.2.3.4", 1234, last_seen=time.time())
    assert isinstance(peer.last_seen, float)


# ---------------------------------------------------------------------------
# Group 2 — MeshDiscovery initialization
# ---------------------------------------------------------------------------


@requires_liboqs
def test_mesh_discovery_init_stores_identity(identity_a):
    MeshDiscovery, _, _ = _import()
    disc = MeshDiscovery(identity_a)
    assert disc._identity is identity_a


@requires_liboqs
def test_mesh_discovery_init_peers_empty(identity_a):
    MeshDiscovery, _, _ = _import()
    disc = MeshDiscovery(identity_a)
    assert disc.get_peers() == {}


@requires_liboqs
async def test_mesh_discovery_context_manager_returns_self(identity_a):
    MeshDiscovery, _, _ = _import()
    disc = MeshDiscovery(identity_a)
    # Context manager without start — just checks __aenter__/__aexit__ exist
    result = await disc.__aenter__()
    assert result is disc
    await disc.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Group 3 — start() / stop() lifecycle
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_start_registers_mdns_service(identity_a):
    MeshDiscovery, _, _ = _import()

    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock()
    mock_azc.async_close = AsyncMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)
        mock_azc.async_register_service.assert_called_once()
        # Verify the ServiceInfo type was passed
        args = mock_azc.async_register_service.call_args[0]
        assert len(args) == 1
        await disc.stop()


@requires_liboqs
async def test_start_creates_service_browser_with_correct_type(identity_a):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    browser_calls = []

    def capture_browser(zc_obj, svc_type, listener):
        browser_calls.append((zc_obj, svc_type, listener))
        return MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser", side_effect=capture_browser):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)
        assert len(browser_calls) == 1
        assert browser_calls[0][1] == SERVICE_TYPE
        await disc.stop()


@requires_liboqs
async def test_stop_unregisters_service(identity_a):
    MeshDiscovery, _, _ = _import()

    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock()
    mock_azc.async_unregister_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)
        await disc.stop()
        mock_azc.async_unregister_service.assert_called_once()


@requires_liboqs
async def test_stop_closes_zeroconf(identity_a):
    MeshDiscovery, _, _ = _import()

    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock()
    mock_azc.async_unregister_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)
        await disc.stop()
        mock_azc.async_close.assert_called_once()


# ---------------------------------------------------------------------------
# Group 4 — TXT record construction
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_txt_record_id_is_16_hex_chars_no_hyphens(identity_a):
    MeshDiscovery, _, _ = _import()

    captured = {}
    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock(
        side_effect=lambda info: captured.update({"info": info})
    )
    mock_azc.async_unregister_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)

    info = captured["info"]
    id_val = info.properties[b"id"].decode()
    assert len(id_val) == 16
    assert "-" not in id_val
    assert id_val == identity_a.device_id.replace("-", "")[:16]
    await disc.stop()


@requires_liboqs
async def test_txt_record_rank_matches_identity(identity_a):
    MeshDiscovery, _, _ = _import()

    captured = {}
    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock(
        side_effect=lambda info: captured.update({"info": info})
    )
    mock_azc.async_unregister_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)

    info = captured["info"]
    assert info.properties[b"rank"] == str(int(identity_a.rank)).encode()
    await disc.stop()


@requires_liboqs
async def test_txt_record_ver_is_1(identity_a):
    MeshDiscovery, _, _ = _import()

    captured = {}
    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock(
        side_effect=lambda info: captured.update({"info": info})
    )
    mock_azc.async_unregister_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)

    assert captured["info"].properties[b"ver"] == b"1"
    await disc.stop()


@requires_liboqs
async def test_txt_record_contains_no_private_key_material(identity_a):
    MeshDiscovery, _, _ = _import()

    captured = {}
    mock_azc = AsyncMock()
    mock_azc.async_register_service = AsyncMock(
        side_effect=lambda info: captured.update({"info": info})
    )
    mock_azc.async_unregister_service = AsyncMock()
    mock_azc.async_close = AsyncMock()
    mock_azc.zeroconf = MagicMock()

    with patch("mesh.discovery.AsyncZeroconf", return_value=mock_azc), \
         patch("mesh.discovery.AsyncServiceBrowser"):
        disc = MeshDiscovery(identity_a)
        await disc.start(port=7777)

    all_txt = b"".join(v for v in captured["info"].properties.values())
    sk_hex = identity_a.signing_keypair.signing_key.raw.hex().encode()
    xk_hex = identity_a.encrypt_keypair.private_key.raw.hex().encode()
    assert sk_hex not in all_txt
    assert xk_hex not in all_txt
    await disc.stop()


# ---------------------------------------------------------------------------
# Group 5 — Peer discovery (add_service)
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_peer_discovered_callback_called_on_new_peer(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    discovered = []
    disc.on_peer_discovered(lambda p: discovered.append(p))

    mock_info = _mock_service_info(identity_b.device_id, identity_b.rank)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(discovered) == 1


@requires_liboqs
async def test_peer_info_fields_set_correctly_from_service_info(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    discovered = []
    disc.on_peer_discovered(lambda p: discovered.append(p))

    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD, addr="192.168.1.5", port=7777)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    peer = discovered[0]
    assert peer.addr == "192.168.1.5"
    assert peer.port == 7777
    assert peer.rank == Rank.SQUAD
    assert peer.device_id == identity_b.device_id.replace("-", "")[:16]


@requires_liboqs
async def test_own_device_id_ignored(identity_a):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    discovered = []
    disc.on_peer_discovered(lambda p: discovered.append(p))

    # Simulate ourselves appearing in mDNS
    mock_info = _mock_service_info(identity_a.device_id, identity_a.rank)
    svc_name = f"ALPHA-1.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(discovered) == 0
    assert disc.get_peers() == {}


@requires_liboqs
async def test_service_info_request_failure_silently_dropped(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    discovered = []
    disc.on_peer_discovered(lambda p: discovered.append(p))

    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    mock_info.async_request = AsyncMock(return_value=False)  # peer not responding
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(discovered) == 0


@requires_liboqs
async def test_missing_txt_key_silently_dropped(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    discovered = []
    disc.on_peer_discovered(lambda p: discovered.append(p))

    mock_info = MagicMock()
    mock_info.async_request = AsyncMock(return_value=True)
    mock_info.parsed_addresses.return_value = ["192.168.1.5"]
    mock_info.port = 7777
    mock_info.properties = {b"rank": b"2", b"ver": b"1"}  # missing 'id'
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(discovered) == 0


@requires_liboqs
async def test_wrong_ver_silently_dropped(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    discovered = []
    disc.on_peer_discovered(lambda p: discovered.append(p))

    mock_info = MagicMock()
    mock_info.async_request = AsyncMock(return_value=True)
    mock_info.parsed_addresses.return_value = ["192.168.1.5"]
    mock_info.port = 7777
    mock_info.properties = {
        b"id": b"abcd1234abcd1234",
        b"rank": b"2",
        b"ver": b"9",  # wrong version
    }
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(discovered) == 0


# ---------------------------------------------------------------------------
# Group 6 — Peer re-advertisement (last_seen update)
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_last_seen_updated_on_readvertisement(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)
        first_seen = list(disc.get_peers().values())[0].last_seen

        # Small delay to get a different timestamp
        await asyncio.sleep(0.05)
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    peers = disc.get_peers()
    assert len(peers) == 1
    assert list(peers.values())[0].last_seen >= first_seen


@requires_liboqs
async def test_discovered_callback_not_called_again_on_update(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    call_count = [0]
    disc.on_peer_discovered(lambda p: call_count.__setitem__(0, call_count[0] + 1))

    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Group 7 — Peer lost (remove_service)
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_peer_lost_callback_called_on_service_removal(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    lost_ids = []
    disc.on_peer_lost(lambda dev_id: lost_ids.append(dev_id))

    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    await disc._on_service_removed(MagicMock(), SERVICE_TYPE, svc_name)
    assert len(lost_ids) == 1
    assert isinstance(lost_ids[0], str)


@requires_liboqs
async def test_peer_removed_from_cache_on_lost(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(disc.get_peers()) == 1
    await disc._on_service_removed(MagicMock(), SERVICE_TYPE, svc_name)
    assert disc.get_peers() == {}


@requires_liboqs
async def test_remove_service_for_unknown_peer_no_crash(identity_a):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    # Should not raise even though peer was never discovered
    await disc._on_service_removed(MagicMock(), SERVICE_TYPE, f"GHOST-9.{SERVICE_TYPE}")


# ---------------------------------------------------------------------------
# Group 8 — update_peer_bundle
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_update_peer_bundle_sets_bundle(identity_a, identity_b, bundle_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"
    peer_id = identity_b.device_id.replace("-", "")[:16]

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    disc.update_peer_bundle(peer_id, bundle_b)
    assert disc.get_peers()[peer_id].public_bundle is bundle_b


@requires_liboqs
async def test_update_peer_bundle_unknown_device_id_no_crash(identity_a, bundle_b):
    MeshDiscovery, _, _ = _import()
    disc = MeshDiscovery(identity_a)
    # Should not raise
    disc.update_peer_bundle("nonexistent0000000", bundle_b)


# ---------------------------------------------------------------------------
# Group 9 — get_peers() isolation
# ---------------------------------------------------------------------------


@requires_liboqs
async def test_get_peers_returns_copy(identity_a, identity_b):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    disc = MeshDiscovery(identity_a)
    mock_info = _mock_service_info(identity_b.device_id, Rank.SQUAD)
    svc_name = f"BRAVO-2.{SERVICE_TYPE}"

    with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
        await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    peers = disc.get_peers()
    peers.clear()  # mutate the returned dict
    assert len(disc.get_peers()) == 1  # internal dict unchanged


@requires_liboqs
async def test_get_peers_returns_all_known_peers(identity_a):
    MeshDiscovery, _, _ = _import()
    from mesh.discovery import SERVICE_TYPE

    id_b = create_identity("BRAVO-2", Rank.SQUAD)
    id_c = create_identity("CHARLIE-3", Rank.OFFICER)

    disc = MeshDiscovery(identity_a)

    for ident, cs in [(id_b, "BRAVO-2"), (id_c, "CHARLIE-3")]:
        mock_info = _mock_service_info(ident.device_id, ident.rank, addr=f"192.168.1.{cs[-1]}")
        svc_name = f"{cs}.{SERVICE_TYPE}"
        with patch("mesh.discovery.ServiceInfo", return_value=mock_info):
            await disc._on_service_added(MagicMock(), SERVICE_TYPE, svc_name)

    assert len(disc.get_peers()) == 2


# ---------------------------------------------------------------------------
# Group 10 — _parse_txt_properties
# ---------------------------------------------------------------------------


@requires_liboqs
def test_parse_txt_invalid_rank_returns_none():
    _, _, _parse_txt_properties = _import()
    result = _parse_txt_properties({b"id": b"abcd1234abcd1234", b"rank": b"9", b"ver": b"1"})
    assert result is None


@requires_liboqs
def test_parse_txt_all_valid_returns_tuple():
    _, _, _parse_txt_properties = _import()
    result = _parse_txt_properties({b"id": b"abcd1234abcd1234", b"rank": b"3", b"ver": b"1"})
    assert result is not None
    device_id_prefix, rank = result
    assert device_id_prefix == "abcd1234abcd1234"
    assert rank == Rank.NCO


@requires_liboqs
def test_parse_txt_non_numeric_rank_returns_none():
    _, _, _parse_txt_properties = _import()
    result = _parse_txt_properties({b"id": b"abcd1234abcd1234", b"rank": b"X", b"ver": b"1"})
    assert result is None
