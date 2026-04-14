"""
cli/mesim_cli.py — MESIM terminal interface

Entry point for the MESIM field node.  Handles identity management,
starts the mesh transport + router, and runs an interactive REPL.

Usage:
  # Create a new identity
  python -m cli.mesim_cli --create ALPHA-1 --rank NCO --identity alpha.json

  # Run a node (prompts for passphrase if --passphrase is omitted)
  python -m cli.mesim_cli --identity alpha.json [--port 7777] [--api-port 8080]

REPL commands:
  peers              — list active encrypted sessions
  routes             — show BATMAN route table
  history <peer_id>  — show last 20 messages with a peer
  send <peer_id> <message>
  whoami             — show own identity
  help / ?           — list commands
  quit / exit / q    — shut down
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import getpass
import hashlib
import hmac
import io
import sys
import time
from pathlib import Path

import uvicorn
from rich.console import Console
from rich.table import Table

from api.server import create_app
from core.crypto import Ed25519VerifyKey, MLKEMPublicKey, SessionKey, X25519PubKey
from core.identity import (
    DeviceIdentity,
    PublicBundle,
    Rank,
    create_identity,
    load_identity,
    save_identity,
)
from core.store import MessageStore
from mesh.discovery import MeshDiscovery
from mesh.router import MeshRouter
from mesh.store_forward import ForwardQueue, StoreForward
from mesh.transport import MeshTransport


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="mesim",
        description="MESIM — defense-grade encrypted P2P mesh communication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--identity", "-i",
        default="identity.json",
        metavar="PATH",
        help="Path to identity file",
    )
    parser.add_argument(
        "--passphrase", "-p",
        metavar="PHRASE",
        default=None,
        help="Identity passphrase (prompted interactively if omitted)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7777,
        metavar="PORT",
        help="UDP mesh listen port",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8080,
        metavar="PORT",
        help="REST API listen port",
    )
    parser.add_argument(
        "--create",
        metavar="CALLSIGN",
        default=None,
        help="Create a new identity with this callsign and exit",
    )
    parser.add_argument(
        "--rank",
        metavar="RANK",
        default="SQUAD",
        help="Rank for --create: COMMAND, OFFICER, NCO, SQUAD",
    )
    parser.add_argument(
        "--lock-timeout",
        type=int,
        default=None,
        metavar="MIN",
        help="Minutes of inactivity before requiring passphrase re-entry (disabled by default)",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rank_from_str(rank_str: str) -> Rank:
    """
    Parse a rank name (case-insensitive) to a Rank enum.

    Raises:
        ValueError: if the name is not a valid rank.
    """
    try:
        return Rank[rank_str.upper()]
    except KeyError:
        valid = ", ".join(r.name for r in Rank)
        raise ValueError(f"Invalid rank {rank_str!r}. Valid values: {valid}")


def load_or_create_identity(
    path: str,
    passphrase: str,
    callsign: str | None = None,
    rank_str: str = "SQUAD",
) -> DeviceIdentity:
    """
    Load an existing identity from *path*, or create a new one.

    A new identity is created when *callsign* is provided AND *path* does
    not already exist.  In all other cases the file is loaded and decrypted
    with *passphrase*.

    Raises:
        ValueError:         if *callsign* is invalid or *rank_str* unknown.
        FileNotFoundError:  if *path* doesn't exist and *callsign* is None.
        InvalidTag:         if *passphrase* is wrong or the file is corrupt.
    """
    id_path = Path(path)
    if callsign is not None and not id_path.exists():
        rank = _rank_from_str(rank_str)
        identity = create_identity(callsign, rank)
        save_identity(identity, path, passphrase)
        return identity
    return load_identity(path, passphrase)


def format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as HH:MM:SS local time."""
    return time.strftime("%H:%M:%S", time.localtime(ts))


def format_message_line(direction: str, callsign: str, text: str, ts: float) -> str:
    """
    Render a single message as a terminal line.

    Sent:     [HH:MM:SS] YOU → CALLSIGN: text
    Received: [HH:MM:SS] CALLSIGN → YOU: text
    """
    t = format_timestamp(ts)
    if direction == "sent":
        return f"[{t}] YOU → {callsign}: {text}"
    return f"[{t}] {callsign} → YOU: {text}"


# ---------------------------------------------------------------------------
# MesimCLI — async interactive REPL
# ---------------------------------------------------------------------------


class MesimCLI:
    """
    Async interactive REPL for a running MESIM node.

    All subsystems are injected so the class is fully unit-testable without
    live networking.  Pass a ``console`` to redirect output (handy in tests).
    """

    def __init__(
        self,
        identity: DeviceIdentity,
        transport: "MeshTransport",
        router: "MeshRouter",
        store_forward: "StoreForward",
        message_store: "MessageStore",
        console: Console | None = None,
        lock_timeout: int | None = None,
        passphrase: str | None = None,
    ) -> None:
        self._identity = identity
        self._transport = transport
        self._router = router
        self._sf = store_forward
        self._store = message_store
        self._console = console or Console()
        self._running = False
        # Dead man's switch: lock after N minutes of inactivity.
        self._lock_timeout = lock_timeout          # None = disabled
        self._last_activity: float = time.time()
        # HMAC key stored in memory; used to verify re-entered passphrase without
        # keeping the plaintext passphrase around any longer than necessary.
        self._passphrase_hmac: bytes | None = (
            self._hash_passphrase(passphrase) if passphrase is not None else None
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Register callbacks and print the welcome banner."""
        self._running = True
        self._transport.on_message(self._on_message_received)
        self._print_banner()

    async def stop(self) -> None:
        self._running = False

    # ── Lock helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hash_passphrase(passphrase: str) -> bytes:
        """Return HMAC-SHA256 of the passphrase under a fixed key for comparison."""
        return hmac.new(b"mesim-lock-v1", passphrase.encode("utf-8"), "sha256").digest()

    def _is_locked(self) -> bool:
        """Return True if the inactivity timeout has elapsed."""
        if self._lock_timeout is None or self._passphrase_hmac is None:
            return False
        elapsed_minutes = (time.time() - self._last_activity) / 60
        return elapsed_minutes >= self._lock_timeout

    async def _prompt_unlock(self) -> None:
        """
        Block the REPL until the correct passphrase is entered.

        Reads from stdin in a thread executor so the event loop stays live
        (incoming mesh packets continue to be processed while locked).
        """
        self._console.print(
            "\n[LOCKED] Inactivity timeout reached. Enter passphrase to unlock:",
            markup=False,
        )
        loop = asyncio.get_running_loop()
        while True:
            entered = await loop.run_in_executor(None, getpass.getpass, "Passphrase: ")
            if hmac.compare_digest(
                self._hash_passphrase(entered),
                self._passphrase_hmac,
            ):
                self._last_activity = time.time()
                self._console.print("Unlocked.", markup=False)
                return
            self._console.print("Incorrect passphrase. Try again.", markup=False)

    # ── Transport callback ─────────────────────────────────────────────────

    async def _on_message_received(self, peer_id: str, payload: bytes) -> None:
        """Called by MeshTransport when an authenticated message arrives."""
        self._store.save_message(peer_id, "recv", payload)
        session = self._transport.get_sessions().get(peer_id)
        callsign = session.peer_bundle.callsign if session else peer_id[:8]
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = f"<binary {len(payload)} bytes>"
        line = format_message_line("recv", callsign, text, time.time())
        self._console.print(f"\n{line}")
        self._console.print("mesim> ", end="")

    # ── Command dispatch ───────────────────────────────────────────────────

    async def handle_command(self, line: str) -> bool:
        """
        Parse and execute one REPL command.

        Returns:
            False if the user asked to quit, True otherwise.
        """
        parts = line.strip().split(None, 2)
        if not parts:
            return True

        self._last_activity = time.time()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            return False

        elif cmd == "peers":
            self._cmd_peers()

        elif cmd == "routes":
            self._cmd_routes()

        elif cmd == "history":
            peer_id = parts[1] if len(parts) >= 2 else None
            self._cmd_history(peer_id)

        elif cmd == "send":
            if len(parts) < 3:
                self._console.print("  Usage: send <peer_id|prefix> <message>")
            else:
                await self._cmd_send(parts[1], parts[2])

        elif cmd == "whoami":
            i = self._identity
            self._console.print(
                f"  {i.callsign} ({i.rank.name})  [{i.device_id}]"
            )

        elif cmd in ("help", "?"):
            self._console.print(
                "  Commands:\n"
                "    peers                        — list active peer sessions (shows full IDs)\n"
                "    routes                       — show BATMAN route table\n"
                "    history <peer_id>            — show message history\n"
                "    send <peer_id|prefix> <msg>  — send; prefix ≥ 8 chars resolves to session\n"
                "    whoami                       — show own identity and device ID\n"
                "    quit                         — exit"
            )

        else:
            self._console.print(f"  Unknown command: {cmd!r}  (type 'help')")

        return True

    # ── Sub-commands ───────────────────────────────────────────────────────

    def _cmd_peers(self) -> None:
        sessions = self._transport.get_sessions()
        if not sessions:
            self._console.print("  (no active peers)")
            return
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Callsign")
        tbl.add_column("Rank")
        tbl.add_column("Peer ID")
        tbl.add_column("Address")
        tbl.add_column("Up (s)")
        tbl.add_column("↑B / ↓B")
        for pid, s in sessions.items():
            age = f"{time.time() - s.established_at:.0f}"
            tbl.add_row(
                s.peer_bundle.callsign,
                s.peer_bundle.rank.name,
                pid,
                f"{s.peer_addr[0]}:{s.peer_addr[1]}",
                age,
                f"{s.bytes_sent}/{s.bytes_recv}",
            )
        self._console.print(tbl)

    def _cmd_routes(self) -> None:
        table = self._router.get_route_table()
        if not table:
            self._console.print("  (no routes)")
            return
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Destination (prefix)")
        tbl.add_column("Via (prefix)")
        tbl.add_column("Hops")
        tbl.add_column("Seq")
        for entry in table.values():
            tbl.add_row(
                entry.originator_id[:16] + "…",
                entry.next_hop_id[:16] + "…",
                str(entry.hop_count),
                str(entry.seq_num),
            )
        self._console.print(tbl)

    def _cmd_history(self, peer_id: str | None) -> None:
        if peer_id is None:
            self._console.print("  Usage: history <peer_id>")
            return
        msgs = self._store.get_history(peer_id, limit=20)
        if not msgs:
            self._console.print("  (no messages)")
            return
        session = self._transport.get_sessions().get(peer_id)
        callsign = session.peer_bundle.callsign if session else peer_id[:8]
        for m in msgs:
            try:
                text = m.plaintext.decode("utf-8")
            except UnicodeDecodeError:
                text = f"<binary {len(m.plaintext)} bytes>"
            self._console.print(format_message_line(m.direction, callsign, text, m.timestamp))

    async def _cmd_send(self, peer_id: str, message: str) -> None:
        payload = message.encode("utf-8")

        # Resolve a peer_id prefix to the full UUID4 session key.
        # Exact match wins first; prefix matching requires ≥ 8 chars.
        sessions = self._transport.get_sessions()
        if peer_id not in sessions:
            if len(peer_id) >= 8:
                matches = [pid for pid in sessions if pid.startswith(peer_id)]
                if len(matches) == 1:
                    peer_id = matches[0]
                elif len(matches) > 1:
                    self._console.print(
                        f"  Ambiguous prefix {peer_id!r} — be more specific:"
                    )
                    for m in matches:
                        self._console.print(f"    {m}")
                    return
                # else: no active session — fall through, let store-forward queue it

        try:
            delivered = await self._sf.send_or_queue(peer_id, payload)
        except Exception as exc:
            self._console.print(f"  Error: {exc}")
            return
        self._store.save_message(peer_id, "sent", payload)
        status = "delivered" if delivered else "queued (peer offline)"
        self._console.print(f"  {status}")

    # ── REPL loop ──────────────────────────────────────────────────────────

    async def run_repl(self) -> None:
        """
        Read commands from stdin until EOF or the user quits.

        Runs stdin reads in a thread executor so the asyncio event loop
        stays unblocked and can handle incoming mesh packets concurrently.
        """
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                if self._is_locked():
                    await self._prompt_unlock()
                self._console.print("mesim> ", end="")
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:  # EOF
                    break
                if not await self.handle_command(line):
                    break
            except (EOFError, KeyboardInterrupt):
                break
        self._console.print("\nGoodbye.")

    # ── Banner ─────────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        i = self._identity
        self._console.print(
            f"[bold]MESIM v1.0[/bold] — {i.callsign} ({i.rank.name}) "
            f"[dim][{i.device_id[:8]}…][/dim]"
        )
        self._console.print("Type [bold]help[/bold] for available commands.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _derive_queue_key(passphrase: str) -> SessionKey:
    """Deterministic 32-byte ForwardQueue encryption key derived from passphrase."""
    return SessionKey(raw=hashlib.sha256(passphrase.encode()).digest())


def _db_paths(identity_path: str) -> tuple[str, str]:
    """Return (msgs_db_path, fwd_db_path) co-located with the identity file."""
    base = str(Path(identity_path).with_suffix(""))
    return base + ".msgs.db", base + ".fwd.db"


def _get_passphrase(args_passphrase: str | None) -> str:
    """Return the passphrase from args or prompt the user securely."""
    if args_passphrase is not None:
        return args_passphrase
    return getpass.getpass("Passphrase: ")


async def _run_node(
    identity: DeviceIdentity,
    passphrase: str,
    port: int,
    api_port: int,
    identity_path: str,
    console: Console,
    lock_timeout: int | None = None,
) -> None:
    """Fully-wired MESIM node: starts all subsystems concurrently, runs until REPL exits."""
    msgs_db, fwd_db = _db_paths(identity_path)
    loop = asyncio.get_running_loop()

    # ── Open sync storage ───────────────────────────────────────────────────
    message_store = MessageStore(msgs_db, passphrase)
    message_store.open()

    fwd_queue = ForwardQueue(fwd_db, _derive_queue_key(passphrase))
    fwd_queue.open()
    fwd_queue.purge_expired()   # clean TTL-expired entries from previous run

    # ── Construct subsystems ────────────────────────────────────────────────
    transport     = MeshTransport(identity)
    router        = MeshRouter(identity, transport)
    store_forward = StoreForward(fwd_queue, transport)
    discovery     = MeshDiscovery(identity)
    cli = MesimCLI(
        identity, transport, router, store_forward, message_store,
        console=console,
        lock_timeout=lock_timeout,
        passphrase=passphrase,
    )
    app = create_app(identity, transport, router, store_forward, message_store)

    # ── Wire discovery → transport ──────────────────────────────────────────
    def _on_peer_discovered(peer_info) -> None:
        """Fetch peer's PublicBundle via their REST API, then open a session."""
        if peer_info.api_port <= 0:
            console.print(
                f"\n[dim]Peer {peer_info.device_id[:8]}… discovered "
                f"but no API port in mDNS record — skipping auto-connect[/dim]"
            )
            return

        async def _fetch_and_connect() -> None:
            import aiohttp  # lazy import; aiohttp is optional at module load time
            url = f"http://{peer_info.addr}:{peer_info.api_port}/bundle"
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status != 200:
                            console.print(
                                f"\n[dim]Bundle fetch {url} → HTTP {resp.status}[/dim]"
                            )
                            return
                        data = await resp.json()

                bundle = PublicBundle(
                    device_id=data["device_id"],
                    callsign=data["callsign"],
                    rank=Rank(data["rank"]),
                    verify_key=Ed25519VerifyKey(
                        raw=base64.b64decode(data["verify_key"])
                    ),
                    encrypt_pub=X25519PubKey(
                        raw=base64.b64decode(data["encrypt_pub"])
                    ),
                    kem_pub=MLKEMPublicKey(
                        raw=base64.b64decode(data["kem_pub"])
                    ),
                    bundle_sig=base64.b64decode(data["bundle_sig"]),
                )
                ok = await transport.connect(
                    (peer_info.addr, peer_info.port), bundle
                )
                if not ok:
                    console.print(
                        f"\n[dim]Handshake with {peer_info.addr}:{peer_info.port} "
                        f"timed out[/dim]"
                    )
            except Exception as exc:
                console.print(
                    f"\n[dim]Bundle fetch failed for "
                    f"{peer_info.addr}:{peer_info.api_port}: {exc}[/dim]"
                )

        asyncio.run_coroutine_threadsafe(_fetch_and_connect(), loop)

    def _on_peer_lost(device_id_prefix: str) -> None:
        console.print(
            f"\n[dim]Peer lost: {device_id_prefix[:8]}… "
            f"(session kept for store-forward)[/dim]"
        )

    discovery.on_peer_discovered(_on_peer_discovered)
    discovery.on_peer_lost(_on_peer_lost)

    # ── Start all subsystems ────────────────────────────────────────────────
    await transport.start("0.0.0.0", port)
    await router.start()
    await store_forward.start()
    await discovery.start(port, api_port)   # advertises self + api_port via mDNS
    await cli.start()             # registers on_message callback, prints banner

    # ── Uvicorn server (embedded in our event loop) ─────────────────────────
    uv_config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=api_port,
        log_level="warning",
        access_log=False,
    )
    uv_server = uvicorn.Server(uv_config)

    # ── Shutdown signal shared between the two concurrent coroutines ─────────
    shutdown_event = asyncio.Event()

    async def _run_repl_then_signal() -> None:
        """Run the interactive REPL; signal all tasks to stop when it exits."""
        try:
            await cli.run_repl()
        finally:
            shutdown_event.set()
            uv_server.should_exit = True

    async def _run_uvicorn_until_shutdown() -> None:
        """Serve the FastAPI app until the shutdown event fires."""
        uv_task = asyncio.create_task(uv_server.serve())
        await shutdown_event.wait()
        uv_server.should_exit = True
        await uv_task

    # ── Run concurrently via asyncio.gather() ───────────────────────────────
    # discovery / transport / router / store_forward manage their own internal
    # background tasks (started above).  The two long-running coroutines that
    # need explicit concurrency are the REPL and the uvicorn server.
    try:
        await asyncio.gather(
            _run_repl_then_signal(),
            _run_uvicorn_until_shutdown(),
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        shutdown_event.set()
        uv_server.should_exit = True
    finally:
        # ── Ordered shutdown ────────────────────────────────────────────────
        await cli.stop()
        await discovery.stop()      # send mDNS goodbye before closing transport
        await router.stop()         # cancel originator broadcast loop
        await store_forward.stop()  # no-op; explicit for completeness
        await transport.stop()      # close UDP socket and retry loop

        fwd_queue.purge_expired()
        fwd_queue.close()
        message_store.close()

        console.print("\nMESIM node shutdown. Stay safe.")


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments, create or load identity, and start the node.

    Returns an exit code (0 = success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    console = Console()

    passphrase = _get_passphrase(args.passphrase)

    # ── Create mode ────────────────────────────────────────────────────────
    if args.create:
        try:
            identity = load_or_create_identity(
                path=args.identity,
                passphrase=passphrase,
                callsign=args.create,
                rank_str=args.rank,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            return 1
        console.print(
            f"[green]Identity created:[/green] {identity.callsign} "
            f"({identity.rank.name}) → {args.identity}"
        )
        return 0

    # ── Run mode ───────────────────────────────────────────────────────────
    try:
        identity = load_identity(args.identity, passphrase)
    except FileNotFoundError:
        console.print(
            f"[red]Error:[/red] Identity file not found: {args.identity}\n"
            "Use --create CALLSIGN to create one."
        )
        return 1
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    console.print(
        f"Loaded identity: {identity.callsign} ({identity.rank.name}) "
        f"[{identity.device_id[:8]}…]"
    )
    console.print(
        f"Mesh UDP port: {args.port}  |  API port: {args.api_port}"
    )
    try:
        asyncio.run(
            _run_node(
                identity=identity,
                passphrase=passphrase,
                port=args.port,
                api_port=args.api_port,
                identity_path=args.identity,
                console=console,
                lock_timeout=args.lock_timeout,
            )
        )
    except KeyboardInterrupt:
        pass  # Ctrl+C before REPL loop started; shutdown message printed inside _run_node
    return 0


if __name__ == "__main__":
    sys.exit(main())
