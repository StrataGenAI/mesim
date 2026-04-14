"""
tests/test_store.py — Unit tests for core/store.py (MessageStore)

Groups:
  1. Schema / lifecycle
  2. save_message
  3. Encryption at rest
  4. get_history
  5. mark_delivered
  6. purge_old
  7. Authentication (wrong passphrase)
  8. Persistence (close + re-open)
"""

import os
import sqlite3
import time

import pytest

from core.store import MessageStore, StoredMessage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db):
    s = MessageStore(tmp_db, "test-passphrase")
    s.open()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Group 1 — Schema / lifecycle
# ---------------------------------------------------------------------------


def test_open_creates_db_file(tmp_db):
    s = MessageStore(tmp_db, "pw")
    s.open()
    s.close()
    assert os.path.exists(tmp_db)


def test_open_creates_messages_table(tmp_db):
    s = MessageStore(tmp_db, "pw")
    s.open()
    s.close()
    conn = sqlite3.connect(tmp_db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert "messages" in tables
    assert "meta" in tables


def test_context_manager_closes(tmp_db):
    with MessageStore(tmp_db, "pw") as s:
        assert s._conn is not None
    assert s._conn is None


def test_double_open_is_idempotent(tmp_db):
    s = MessageStore(tmp_db, "pw")
    s.open()
    s.open()  # must not raise
    s.close()


def test_close_without_open_is_safe(tmp_db):
    s = MessageStore(tmp_db, "pw")
    s.close()  # must not raise


# ---------------------------------------------------------------------------
# Group 2 — save_message
# ---------------------------------------------------------------------------


def test_save_message_returns_int(store):
    msg_id = store.save_message("peer1", "sent", b"hello")
    assert isinstance(msg_id, int)
    assert msg_id > 0


def test_save_message_increments_id(store):
    id1 = store.save_message("peer1", "sent", b"first")
    id2 = store.save_message("peer1", "sent", b"second")
    assert id2 > id1


def test_save_message_direction_sent(store):
    msg_id = store.save_message("peer1", "sent", b"outbound")
    history = store.get_history("peer1")
    assert history[0].direction == "sent"


def test_save_message_direction_recv(store):
    store.save_message("peer1", "recv", b"inbound")
    history = store.get_history("peer1")
    assert history[0].direction == "recv"


def test_save_message_invalid_direction_raises(store):
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        store.save_message("peer1", "unknown", b"bad direction")


def test_save_message_uses_current_time_when_none(store):
    before = time.time()
    store.save_message("peer1", "sent", b"ts test")
    after = time.time()
    history = store.get_history("peer1")
    assert before <= history[0].timestamp <= after


def test_save_message_accepts_explicit_timestamp(store):
    ts = 1_000_000.0
    store.save_message("peer1", "sent", b"with ts", timestamp=ts)
    history = store.get_history("peer1")
    assert history[0].timestamp == ts


# ---------------------------------------------------------------------------
# Group 3 — Encryption at rest
# ---------------------------------------------------------------------------


def test_plaintext_not_in_db_file(tmp_db):
    secret = b"TOP SECRET DELTA FORCE"
    with MessageStore(tmp_db, "pw") as s:
        s.save_message("peer1", "sent", secret)

    raw = open(tmp_db, "rb").read()
    assert secret not in raw


def test_different_messages_produce_different_ciphertexts(tmp_db):
    """Each save encrypts with a fresh nonce — same plaintext produces different BLOBs."""
    with MessageStore(tmp_db, "pw") as s:
        s.save_message("peer1", "sent", b"identical")
        s.save_message("peer1", "sent", b"identical")

    conn = sqlite3.connect(tmp_db)
    blobs = [row[0] for row in conn.execute("SELECT ciphertext FROM messages")]
    conn.close()
    assert blobs[0] != blobs[1]


# ---------------------------------------------------------------------------
# Group 4 — get_history
# ---------------------------------------------------------------------------


def test_get_history_empty(store):
    assert store.get_history("nobody") == []


def test_get_history_returns_stored_message_objects(store):
    store.save_message("peer1", "sent", b"hi")
    history = store.get_history("peer1")
    assert len(history) == 1
    assert isinstance(history[0], StoredMessage)


def test_get_history_plaintext_matches(store):
    store.save_message("peer1", "sent", b"check plaintext")
    assert store.get_history("peer1")[0].plaintext == b"check plaintext"


def test_get_history_ordered_oldest_first(store):
    store.save_message("peer1", "sent", b"first",  timestamp=1000.0)
    store.save_message("peer1", "sent", b"second", timestamp=2000.0)
    history = store.get_history("peer1")
    assert history[0].plaintext == b"first"
    assert history[1].plaintext == b"second"


def test_get_history_limit(store):
    for i in range(5):
        store.save_message("peer1", "sent", f"msg{i}".encode())
    history = store.get_history("peer1", limit=3)
    assert len(history) == 3


def test_get_history_before_id_pagination(store):
    ids = [store.save_message("peer1", "sent", f"m{i}".encode()) for i in range(5)]
    # Get messages before the 4th id (should return first 3)
    history = store.get_history("peer1", before_id=ids[3])
    assert all(m.id < ids[3] for m in history)


def test_get_history_peer_isolation(store):
    store.save_message("alice", "sent", b"for alice")
    store.save_message("bob",   "recv", b"for bob")
    assert len(store.get_history("alice")) == 1
    assert store.get_history("alice")[0].plaintext == b"for alice"
    assert len(store.get_history("bob")) == 1


def test_get_history_default_limit_is_100(store):
    for i in range(110):
        store.save_message("peer1", "sent", f"{i}".encode(), timestamp=float(i))
    assert len(store.get_history("peer1")) == 100


# ---------------------------------------------------------------------------
# Group 5 — mark_delivered
# ---------------------------------------------------------------------------


def test_mark_delivered_sets_flag(store):
    msg_id = store.save_message("peer1", "sent", b"deliver me")
    history = store.get_history("peer1")
    assert history[0].delivered is False

    store.mark_delivered(msg_id)
    history = store.get_history("peer1")
    assert history[0].delivered is True


def test_mark_delivered_idempotent(store):
    msg_id = store.save_message("peer1", "sent", b"twice")
    store.mark_delivered(msg_id)
    store.mark_delivered(msg_id)  # must not raise
    assert store.get_history("peer1")[0].delivered is True


def test_mark_delivered_unknown_id_is_noop(store):
    store.mark_delivered(999999)  # must not raise


# ---------------------------------------------------------------------------
# Group 6 — purge_old
# ---------------------------------------------------------------------------


def test_purge_old_deletes_old_rows(store):
    old_ts = time.time() - 8 * 86400  # 8 days ago
    store.save_message("peer1", "sent", b"old msg", timestamp=old_ts)
    deleted = store.purge_old()
    assert deleted == 1
    assert store.get_history("peer1") == []


def test_purge_old_keeps_recent(store):
    store.save_message("peer1", "sent", b"fresh")
    deleted = store.purge_old()
    assert deleted == 0
    assert len(store.get_history("peer1")) == 1


def test_purge_old_returns_count(store):
    old_ts = time.time() - 8 * 86400
    for _ in range(3):
        store.save_message("peer1", "sent", b"old", timestamp=old_ts)
    assert store.purge_old() == 3


def test_purge_old_custom_threshold(store):
    ts = time.time() - 3600  # 1 hour ago
    store.save_message("peer1", "sent", b"hour old", timestamp=ts)
    # 7-day default → not deleted
    assert store.purge_old(max_age_seconds=7 * 86400) == 0
    # 30-minute threshold → deleted
    assert store.purge_old(max_age_seconds=1800) == 1


# ---------------------------------------------------------------------------
# Group 7 — Authentication (wrong passphrase)
# ---------------------------------------------------------------------------


def test_wrong_passphrase_on_reopen_raises(tmp_db):
    with MessageStore(tmp_db, "correct-pw") as s:
        s.save_message("peer1", "sent", b"secret")

    with pytest.raises(Exception):  # InvalidTag or ValueError
        with MessageStore(tmp_db, "wrong-pw") as s:
            pass


def test_correct_passphrase_on_reopen_works(tmp_db):
    with MessageStore(tmp_db, "correct-pw") as s:
        s.save_message("peer1", "sent", b"secret")

    with MessageStore(tmp_db, "correct-pw") as s:
        history = s.get_history("peer1")
    assert history[0].plaintext == b"secret"


# ---------------------------------------------------------------------------
# Group 8 — Persistence (close + re-open)
# ---------------------------------------------------------------------------


def test_persistence_survives_close_reopen(tmp_db):
    with MessageStore(tmp_db, "pw") as s:
        s.save_message("peer1", "sent", b"persist me")

    with MessageStore(tmp_db, "pw") as s:
        history = s.get_history("peer1")
    assert len(history) == 1
    assert history[0].plaintext == b"persist me"


def test_persistence_multiple_messages(tmp_db):
    with MessageStore(tmp_db, "pw") as s:
        for i in range(5):
            s.save_message("peer1", "sent", f"msg{i}".encode(), timestamp=float(i))

    with MessageStore(tmp_db, "pw") as s:
        history = s.get_history("peer1")
    assert len(history) == 5
    assert [m.plaintext for m in history] == [f"msg{i}".encode() for i in range(5)]


def test_persistence_delivered_flag_survives(tmp_db):
    with MessageStore(tmp_db, "pw") as s:
        msg_id = s.save_message("peer1", "sent", b"mark me")
        s.mark_delivered(msg_id)

    with MessageStore(tmp_db, "pw") as s:
        history = s.get_history("peer1")
    assert history[0].delivered is True
