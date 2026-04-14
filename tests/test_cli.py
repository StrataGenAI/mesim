"""
tests/test_cli.py — MESIM CLI logic tests

Tests helper functions and MesimCLI command dispatch.  No networking or
live crypto is needed — subsystems are replaced with lightweight mocks.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from cli.mesim_cli import (
    MesimCLI,
    _rank_from_str,
    build_arg_parser,
    format_message_line,
    format_timestamp,
    load_or_create_identity,
)
from core.identity import Rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_console() -> tuple[Console, io.StringIO]:
    """Return a (Console, buffer) pair so tests can inspect output."""
    buf = io.StringIO()
    console = Console(file=buf, highlight=False, markup=False)
    return console, buf


def _make_identity_mock(
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
    bytes_sent: int = 0,
    bytes_recv: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.peer_id = peer_id
    s.peer_addr = (host, port)
    s.peer_bundle = _make_bundle(callsign, rank)
    s.established_at = established_at
    s.bytes_sent = bytes_sent
    s.bytes_recv = bytes_recv
    return s


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


def _make_route(
    originator_id: str = "cc00" * 8,
    next_hop_id: str = "ff00" * 8,
    hop_count: int = 2,
    seq_num: int = 5,
) -> MagicMock:
    e = MagicMock()
    e.originator_id = originator_id
    e.next_hop_id = next_hop_id
    e.hop_count = hop_count
    e.seq_num = seq_num
    return e


def _make_cli(
    *,
    identity=None,
    sessions: dict | None = None,
    route_table: dict | None = None,
    send_result: bool = True,
    send_raises: Exception | None = None,
    history: list | None = None,
    save_result: int = 1,
) -> tuple[MesimCLI, io.StringIO]:
    """Construct a MesimCLI with mocked subsystems and return (cli, output_buf)."""
    ident = identity or _make_identity_mock()
    sessions = sessions if sessions is not None else {}
    route_table = route_table if route_table is not None else {}
    history = history if history is not None else []

    transport = MagicMock()
    transport.get_sessions.return_value = sessions
    transport.on_message = MagicMock()

    router = MagicMock()
    router.get_route_table.return_value = route_table

    sf = MagicMock()
    if send_raises is not None:
        sf.send_or_queue = AsyncMock(side_effect=send_raises)
    else:
        sf.send_or_queue = AsyncMock(return_value=send_result)

    ms = MagicMock()
    ms.get_history.return_value = history
    ms.save_message.return_value = save_result

    console, buf = _make_console()
    cli = MesimCLI(ident, transport, router, sf, ms, console=console)
    return cli, buf


# ---------------------------------------------------------------------------
# build_arg_parser
# ---------------------------------------------------------------------------


class TestBuildArgParser:
    def test_returns_parser(self) -> None:
        import argparse
        p = build_arg_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_default_identity_path(self) -> None:
        args = build_arg_parser().parse_args([])
        assert args.identity == "identity.json"

    def test_default_port(self) -> None:
        args = build_arg_parser().parse_args([])
        assert args.port == 7777

    def test_default_api_port(self) -> None:
        args = build_arg_parser().parse_args([])
        assert args.api_port == 8080

    def test_default_rank(self) -> None:
        args = build_arg_parser().parse_args([])
        assert args.rank == "SQUAD"

    def test_custom_identity(self) -> None:
        args = build_arg_parser().parse_args(["--identity", "my.json"])
        assert args.identity == "my.json"

    def test_custom_port(self) -> None:
        args = build_arg_parser().parse_args(["--port", "9999"])
        assert args.port == 9999

    def test_create_flag(self) -> None:
        args = build_arg_parser().parse_args(["--create", "ECHO-5"])
        assert args.create == "ECHO-5"

    def test_rank_flag(self) -> None:
        args = build_arg_parser().parse_args(["--rank", "OFFICER"])
        assert args.rank == "OFFICER"

    def test_passphrase_flag(self) -> None:
        args = build_arg_parser().parse_args(["--passphrase", "secret"])
        assert args.passphrase == "secret"


# ---------------------------------------------------------------------------
# _rank_from_str
# ---------------------------------------------------------------------------


class TestRankFromStr:
    def test_squad(self) -> None:
        assert _rank_from_str("SQUAD") == Rank.SQUAD

    def test_nco(self) -> None:
        assert _rank_from_str("NCO") == Rank.NCO

    def test_officer(self) -> None:
        assert _rank_from_str("OFFICER") == Rank.OFFICER

    def test_command(self) -> None:
        assert _rank_from_str("COMMAND") == Rank.COMMAND

    def test_case_insensitive(self) -> None:
        assert _rank_from_str("squad") == Rank.SQUAD
        assert _rank_from_str("Nco") == Rank.NCO

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid rank"):
            _rank_from_str("GENERAL")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            _rank_from_str("")


# ---------------------------------------------------------------------------
# load_or_create_identity
# ---------------------------------------------------------------------------


class TestLoadOrCreateIdentity:
    def test_creates_new_identity_file(self, tmp_path: Path) -> None:
        path = tmp_path / "id.json"
        identity = load_or_create_identity(
            str(path), "passphrase", callsign="ALPHA-1", rank_str="SQUAD"
        )
        assert identity.callsign == "ALPHA-1"
        assert identity.rank == Rank.SQUAD
        assert path.exists()

    def test_created_identity_loadable(self, tmp_path: Path) -> None:
        path = tmp_path / "id.json"
        original = load_or_create_identity(
            str(path), "secret", callsign="BRAVO-2", rank_str="NCO"
        )
        loaded = load_or_create_identity(str(path), "secret")
        assert loaded.device_id == original.device_id
        assert loaded.callsign == original.callsign

    def test_loads_existing_when_callsign_given(self, tmp_path: Path) -> None:
        path = tmp_path / "id.json"
        # Create it first
        first = load_or_create_identity(
            str(path), "pw", callsign="CHARLIE", rank_str="SQUAD"
        )
        # Load it with callsign= set (file exists → load, don't overwrite)
        second = load_or_create_identity(str(path), "pw", callsign="DELTA")
        assert second.device_id == first.device_id

    def test_missing_file_without_callsign_raises(self, tmp_path: Path) -> None:
        from cryptography.exceptions import InvalidTag
        path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            load_or_create_identity(str(path), "pw")

    def test_wrong_passphrase_raises(self, tmp_path: Path) -> None:
        from cryptography.exceptions import InvalidTag
        path = tmp_path / "id.json"
        load_or_create_identity(str(path), "correct", callsign="X", rank_str="SQUAD")
        with pytest.raises(InvalidTag):
            load_or_create_identity(str(path), "wrong")

    def test_invalid_callsign_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        with pytest.raises(ValueError):
            load_or_create_identity(str(path), "pw", callsign="bad callsign!")

    def test_invalid_rank_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "rank.json"
        with pytest.raises(ValueError, match="Invalid rank"):
            load_or_create_identity(str(path), "pw", callsign="ECHO", rank_str="ADMIRAL")


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_returns_string(self) -> None:
        assert isinstance(format_timestamp(time.time()), str)

    def test_format_hhmmss(self) -> None:
        # Use a known timestamp for determinism
        ts = time.mktime(time.strptime("2026-04-12 14:30:55", "%Y-%m-%d %H:%M:%S"))
        result = format_timestamp(ts)
        assert result == "14:30:55"

    def test_length_eight(self) -> None:
        result = format_timestamp(time.time())
        assert len(result) == 8

    def test_colon_separators(self) -> None:
        result = format_timestamp(time.time())
        assert result[2] == ":" and result[5] == ":"


# ---------------------------------------------------------------------------
# format_message_line
# ---------------------------------------------------------------------------


class TestFormatMessageLine:
    def test_sent_contains_you(self) -> None:
        line = format_message_line("sent", "BRAVO-2", "hi", 0.0)
        assert "YOU" in line

    def test_sent_contains_callsign(self) -> None:
        line = format_message_line("sent", "BRAVO-2", "hi", 0.0)
        assert "BRAVO-2" in line

    def test_sent_contains_message(self) -> None:
        line = format_message_line("sent", "BRAVO-2", "hello world", 0.0)
        assert "hello world" in line

    def test_recv_callsign_first(self) -> None:
        line = format_message_line("recv", "CHARLIE-3", "msg", 0.0)
        assert "CHARLIE-3" in line
        assert "YOU" in line
        # Callsign should appear before YOU in the recv direction
        assert line.index("CHARLIE-3") < line.index("YOU")

    def test_contains_timestamp(self) -> None:
        ts = time.mktime(time.strptime("2026-04-12 09:00:00", "%Y-%m-%d %H:%M:%S"))
        line = format_message_line("sent", "X", "msg", ts)
        assert "09:00:00" in line

    def test_sent_and_recv_differ(self) -> None:
        ts = time.time()
        sent = format_message_line("sent", "ALPHA", "msg", ts)
        recv = format_message_line("recv", "ALPHA", "msg", ts)
        assert sent != recv


# ---------------------------------------------------------------------------
# MesimCLI.handle_command — quit / empty
# ---------------------------------------------------------------------------


class TestHandleCommandQuit:
    @pytest.mark.asyncio
    async def test_quit_returns_false(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("quit") is False

    @pytest.mark.asyncio
    async def test_exit_returns_false(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("exit") is False

    @pytest.mark.asyncio
    async def test_q_returns_false(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("q") is False

    @pytest.mark.asyncio
    async def test_empty_line_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("") is True

    @pytest.mark.asyncio
    async def test_whitespace_line_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("   ") is True


# ---------------------------------------------------------------------------
# MesimCLI.handle_command — peers
# ---------------------------------------------------------------------------


class TestHandleCommandPeers:
    @pytest.mark.asyncio
    async def test_no_peers_message(self) -> None:
        cli, buf = _make_cli(sessions={})
        await cli.handle_command("peers")
        assert "no active peers" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_with_peer_shows_callsign(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, callsign="BRAVO-2")
        cli, buf = _make_cli(sessions={pid: s})
        await cli.handle_command("peers")
        assert "BRAVO-2" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_peers_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("peers") is True


# ---------------------------------------------------------------------------
# MesimCLI.handle_command — routes
# ---------------------------------------------------------------------------


class TestHandleCommandRoutes:
    @pytest.mark.asyncio
    async def test_no_routes_message(self) -> None:
        cli, buf = _make_cli(route_table={})
        await cli.handle_command("routes")
        assert "no routes" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_with_route_shows_hop_count(self) -> None:
        oid = "cc00" * 8
        e = _make_route(originator_id=oid, hop_count=3)
        cli, buf = _make_cli(route_table={oid: e})
        await cli.handle_command("routes")
        assert "3" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_routes_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("routes") is True


# ---------------------------------------------------------------------------
# MesimCLI.handle_command — history
# ---------------------------------------------------------------------------


class TestHandleCommandHistory:
    @pytest.mark.asyncio
    async def test_missing_peer_id_shows_usage(self) -> None:
        cli, buf = _make_cli()
        await cli.handle_command("history")
        assert "Usage" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_no_messages(self) -> None:
        cli, buf = _make_cli(history=[])
        await cli.handle_command("history ff00ff00")
        assert "no messages" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_shows_message_text(self) -> None:
        msg = _make_stored_message(plaintext=b"classified info")
        cli, buf = _make_cli(history=[msg])
        await cli.handle_command("history ff00ff00")
        assert "classified info" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_binary_payload_shows_binary_label(self) -> None:
        msg = _make_stored_message(plaintext=b"\xff\xfe\xfd")  # invalid UTF-8
        cli, buf = _make_cli(history=[msg])
        await cli.handle_command("history ff00ff00")
        assert "binary" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_callsign_from_session(self) -> None:
        pid = "ff00" * 8
        msg = _make_stored_message(peer_id=pid, plaintext=b"hi")
        s = _make_session(peer_id=pid, callsign="ECHO-9")
        cli, buf = _make_cli(sessions={pid: s}, history=[msg])
        await cli.handle_command(f"history {pid}")
        assert "ECHO-9" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_history_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("history peer") is True


# ---------------------------------------------------------------------------
# MesimCLI.handle_command — send
# ---------------------------------------------------------------------------


class TestHandleCommandSend:
    @pytest.mark.asyncio
    async def test_missing_args_shows_usage(self) -> None:
        cli, buf = _make_cli()
        await cli.handle_command("send")
        assert "Usage" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_missing_message_shows_usage(self) -> None:
        cli, buf = _make_cli()
        await cli.handle_command("send ff00ff00")
        assert "Usage" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_delivered_shows_delivered(self) -> None:
        cli, buf = _make_cli(send_result=True)
        await cli.handle_command("send ff00ff00 hello")
        assert "delivered" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_queued_shows_queued(self) -> None:
        cli, buf = _make_cli(send_result=False)
        await cli.handle_command("send ff00ff00 hello")
        assert "queued" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_error_shows_error(self) -> None:
        cli, buf = _make_cli(send_raises=RuntimeError("queue full"))
        await cli.handle_command("send ff00ff00 hello")
        assert "Error" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_saves_to_message_store(self) -> None:
        cli, _ = _make_cli(send_result=True)
        await cli.handle_command("send ff00ff00 my message")
        cli._store.save_message.assert_called_once_with("ff00ff00", "sent", b"my message")

    @pytest.mark.asyncio
    async def test_send_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("send ff00ff00 hello") is True

    @pytest.mark.asyncio
    async def test_message_with_spaces_sent_whole(self) -> None:
        cli, _ = _make_cli()
        await cli.handle_command("send ff00ff00 one two three")
        cli._store.save_message.assert_called_once_with(
            "ff00ff00", "sent", b"one two three"
        )


# ---------------------------------------------------------------------------
# MesimCLI.handle_command — whoami / help / unknown
# ---------------------------------------------------------------------------


class TestHandleCommandMisc:
    @pytest.mark.asyncio
    async def test_whoami_shows_callsign(self) -> None:
        ident = _make_identity_mock(callsign="FOXTROT-6")
        cli, buf = _make_cli(identity=ident)
        await cli.handle_command("whoami")
        assert "FOXTROT-6" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_whoami_shows_rank(self) -> None:
        ident = _make_identity_mock(rank=Rank.COMMAND)
        cli, buf = _make_cli(identity=ident)
        await cli.handle_command("whoami")
        assert "COMMAND" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_help_lists_commands(self) -> None:
        cli, buf = _make_cli()
        await cli.handle_command("help")
        out = buf.getvalue()
        assert "send" in out
        assert "peers" in out
        assert "routes" in out
        assert "quit" in out

    @pytest.mark.asyncio
    async def test_question_mark_help(self) -> None:
        cli, buf = _make_cli()
        await cli.handle_command("?")
        assert "send" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_unknown_command_message(self) -> None:
        cli, buf = _make_cli()
        await cli.handle_command("frobulate")
        assert "Unknown command" in buf.getvalue() or "frobulate" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_misc_returns_true(self) -> None:
        cli, _ = _make_cli()
        assert await cli.handle_command("whoami") is True


# ---------------------------------------------------------------------------
# MesimCLI._on_message_received
# ---------------------------------------------------------------------------


class TestOnMessageReceived:
    @pytest.mark.asyncio
    async def test_saves_to_store(self) -> None:
        pid = "ff00" * 8
        cli, _ = _make_cli()
        await cli._on_message_received(pid, b"incoming")
        cli._store.save_message.assert_called_once_with(pid, "recv", b"incoming")

    @pytest.mark.asyncio
    async def test_shows_decoded_text(self) -> None:
        pid = "ff00" * 8
        cli, buf = _make_cli()
        await cli._on_message_received(pid, b"hello world")
        assert "hello world" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_binary_payload_shown_as_binary(self) -> None:
        pid = "ff00" * 8
        cli, buf = _make_cli()
        await cli._on_message_received(pid, b"\xff\xfe\xfd")  # invalid UTF-8
        assert "binary" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_callsign_from_session(self) -> None:
        pid = "ff00" * 8
        s = _make_session(peer_id=pid, callsign="HOTEL-8")
        cli, buf = _make_cli(sessions={pid: s})
        await cli._on_message_received(pid, b"hi")
        assert "HOTEL-8" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_fallback_peer_id_prefix_when_no_session(self) -> None:
        pid = "abcdef01" * 4
        cli, buf = _make_cli(sessions={})
        await cli._on_message_received(pid, b"hello")
        # Should show first 8 chars of peer_id as callsign fallback
        assert "abcdef01" in buf.getvalue()
