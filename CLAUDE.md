# CLAUDE.md
# Project: mesim
# Created: 2026-04-11
# Owner: stratagenhq

## Project Overview
Defense-grade encrypted P2P communication platform for military field operations.
Offline-first mesh network, no internet required, up to 15km multi-hop range.

## Tech Stack
- Language: Python 3.13 (3.11+ compatible)
- Networking: asyncio + websockets + zeroconf (mDNS)
- Crypto: cryptography + liboqs-python (ML-KEM-768 PQC)
- Database: SQLite + SQLCipher (encrypted at rest)
- Mesh protocol: BATMAN-inspired routing over UDP
- API: FastAPI (device-to-device REST)
- KDF: Argon2id via argon2-cffi

## Architecture
```
core/       — crypto primitives, identity, message store
mesh/       — mDNS discovery, BATMAN routing, UDP transport
api/        — FastAPI device server
cli/        — terminal interface
tests/      — pytest unit tests
```

## Key Files
- Entry point: cli/mesim_cli.py (Phase 1)
- Crypto: core/crypto.py
- Identity: core/identity.py
- Tests: tests/

## Coding Standards
- Language version: Python 3.11+
- Style guide: PEP 8
- Naming conventions: snake_case
- Error handling: explicit raises, no silent failures, InvalidTag = same error for wrong passphrase or corruption (oracle prevention)

## Workflow
- Test command: `pytest tests/ -v`
- Build command: N/A (pure Python)
- Deploy command: N/A (field deployment TBD)

## MANDATORY GIT DISCIPLINE
After EVERY phase completion, before saying "done":
1. `git add -A`
2. `git commit -m "feat(phaseN): description of what was built"`
3. Confirm commit hash to user

**Never say "X/X tests passing" without committing first.**
**A phase is NOT complete until it is committed.**

## Current Sprint / Active Work
- Phase 1: DONE — encrypted P2P chat between 2 nodes on same WiFi (145 tests passing)
- DONE: core/crypto.py (41 tests)
- DONE: core/identity.py (33 tests)
- DONE: mesh/discovery.py (mDNS)
- DONE: mesh/transport.py (UDP transport, 4-step hybrid KEM handshake, fragmentation, retry)
- Phase 2: DONE (241 tests passing)
  - DONE: core/store.py (34 tests — SQLite + app-level encryption)
  - DONE: mesh/store_forward.py (31 tests — DTN queue, TTL=24h, auto-flush on reconnect)
  - DONE: mesh/router.py (31 tests — BATMAN originator broadcasts, 7-hop max, Ed25519-signed)
- Phase 3: DONE (350 tests passing)
  - DONE: api/server.py (39 tests — FastAPI create_app() factory, /health /peers /routes /send /messages)
  - DONE: cli/mesim_cli.py (70 tests — argparse, rich Console, MesimCLI async REPL, load_or_create_identity)
- Phase 4: DONE (398 tests passing)
  - DONE: mesh/store_forward.py — peer_id normalization (32-char no-hyphen → UUID4), periodic 30s flush loop, console output
  - DONE: api/server.py — rate limiting (100 req/min sliding window per IP, HTTP 429), message size cap (4096 bytes, HTTP 413)
  - DONE: core/identity.py — duress PIN (save_identity duress_passphrase kwarg; load with duress → decoy identity + wipe real keys)
  - DONE: cli/mesim_cli.py — --lock-timeout N (dead man's switch; HMAC-verified passphrase re-entry after N min inactivity)
  - DONE: tests/test_integration.py (3 tests — full E2E two-node: handshake+5 live, queue+reconnect, 5+3=8 across offline/reconnect)

## Known Issues / Gotchas
- Firewall: the following ports must be open between nodes for full operation:
  - UDP mesh ports (e.g. 9001, 9002) — BATMAN routing + encrypted message transport
  - TCP API ports (e.g. 8001, 8002) — bundle fetch (`GET /bundle`) before handshake
  - In production: API ports should be firewalled to LAN/VLAN only, never internet-exposed
  - The API server binds to `0.0.0.0` so peers can reach it at the node's real IP; restrict via iptables/nftables in hardened deployments

## Threat Model
- **Passive observer**: sign-then-encrypt hides sender identity; PQC (ML-KEM-768) defeats harvest-now/decrypt-later
- **Active MITM**: PublicBundle signed over all fields (Ed25519); verified before any KEM exchange; bundle_sig covers device_id
- **Captured device**: Argon2id KDF (64 MiB, t=3) slows brute force; duress PIN wipes real keys and returns decoy identity
- **Physical coercion**: duress passphrase → decoy identity (same callsign/rank, different device_id/keys); real keys erased from disk on first duress load
- **API abuse**: rate limit (100 req/min per IP), message size cap (4096 bytes), API binds 0.0.0.0 but must be restricted via iptables in production
- **Replay/DoS**: originator seq_num loop prevention; MAX_REASSEMBLY_BUFS=64; MAX_QUEUE_PER_PEER=256 in store-forward
- **Oracle**: wrong passphrase and corrupted file both raise InvalidTag (indistinguishable); HMAC-SHA256 sentinel in MessageStore

## Deployment Guide
```
# Create identity
python -m cli.mesim_cli --create ALPHA-1 --rank NCO --identity alpha.json

# Run node (passphrase prompted)
python -m cli.mesim_cli --identity alpha.json --port 9001 --api-port 8001

# Run node with dead man's switch (lock after 10 min inactivity)
python -m cli.mesim_cli --identity alpha.json --lock-timeout 10

# Firewall rules (iptables example)
iptables -A INPUT -p udp --dport 9001 -j ACCEPT   # mesh transport
iptables -A INPUT -p tcp --dport 8001 -s 10.0.0.0/24 -j ACCEPT  # API (LAN only)
iptables -A INPUT -p tcp --dport 8001 -j DROP
```

## Decisions Log
- 2026-04-11: CLAUDE.md created on project init per Stratagen Master System Prompt v2.6
- 2026-04-11: Used `cryptography` package instead of PyNaCl (not installed); full abstraction behind typed dataclasses means zero callers import cryptography directly
- 2026-04-11: ML-KEM-768 public key extracted from secret key at byte offset 1152 per FIPS 203 (dk = dk_PKE[1152] || ek[1184] || H(ek)[32] || z[32]); no oqs call needed at load time
- 2026-04-11: sign-then-encrypt ordering chosen: signature inside ciphertext hides sender from passive observers (sealed sender)
- 2026-04-12: Never call _bundle_with_device_id before verify_public_bundle — bundle_sig covers device_id; overwriting device_id (36-char UUID4) with sender_id header bytes (32-char no-hyphen) before verification corrupts the canonical hash and silently drops all handshake packets
- 2026-04-12: Always pre-compute pkt_header before encrypt_message to use as AAD — payload_len is deterministic (FRAG_HEADER_SIZE + NONCE_SIZE + len(chunk) + TAG_SIZE), so pkt_header can be built first and passed as aad= to both encrypt_message and decrypt_message, ensuring ChaCha20-Poly1305 tag covers the packet header
- 2026-04-12: core/store.py uses stdlib sqlite3 + application-level ChaCha20-Poly1305 instead of SQLCipher — libsqlcipher not installed on target; all sensitive columns stored as nonce||ciphertext BLOBs; _open_db() abstraction allows future SQLCipher swap; key verified via HMAC-SHA256 sentinel on every re-open (wrong passphrase = InvalidTag, oracle-safe)
- 2026-04-12: mesh/router.py originator packets signed over fields excluding hop_count (hop_count changes at each relay); signature covers originator_id||seq_num||max_hops||timestamp so intermediaries can verify and re-forward _repack_originator without re-signing; _repack_originator preserves original signature
- 2026-04-12: RouteEntry.last_seen uses the originator's timestamp (not time.time()) so stale routes can be detected based on when the originator last broadcast, not when we forwarded it
- 2026-04-12: api/server.py uses create_app() factory (not module-level singletons) so all subsystems are injected via app.state — enables full TestClient-based testing without networking
- 2026-04-12: cli/mesim_cli.py accepts an optional rich.Console in MesimCLI.__init__() for test output capture; Console(file=StringIO()) in tests avoids capsys conflicts with rich markup
- 2026-04-12: binary bytes(range(N)) passes UTF-8 decode for N≤127; always use b"\xff\xfe..." for guaranteed-invalid UTF-8 test payloads
- 2026-04-14: mesh/store_forward.py normalize_peer_id() converts 32-char no-hyphen wire format to UUID4; applied at all DB access points so messages queued with old format are always findable
- 2026-04-14: StoreForward periodic flush loop uses module-level _FLUSH_INTERVAL=30; monkeypatched to 0.05 in tests for speed
- 2026-04-14: api/server.py rate limit stored in app.state.rate_buckets dict (client_ip→deque); each create_app() call gets fresh state so test isolation is free
- 2026-04-14: identity.py duress PIN stores decoy keys under separate duress_salt/duress_nonce/duress_encrypted_keys; on duress activation, save_identity(decoy, path, duress_passphrase) overwrites file — duress fields absent in wiped file
- 2026-04-14: CLI lock uses HMAC-SHA256(key=b"mesim-lock-v1", msg=passphrase) not plaintext; hmac.compare_digest() prevents timing attacks on passphrase comparison
