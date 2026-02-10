from types import SimpleNamespace

from nanobot.channels.telegram import TelegramChannel


def _user(user_id: int = 1, username: str | None = None, first_name: str | None = None):
    return SimpleNamespace(id=user_id, username=username, first_name=first_name)


def _message(**kwargs):
    return SimpleNamespace(**kwargs)


def test_extract_reply_context_none():
    prefix, meta, key = TelegramChannel._extract_reply_context(None)
    assert prefix is None
    assert meta == {}
    assert key is None

    msg = _message(reply_to_message=None)
    prefix, meta, key = TelegramChannel._extract_reply_context(msg)
    assert prefix is None
    assert meta == {}
    assert key is None


def test_extract_reply_context_text():
    reply_user = _user(user_id=42, username="alice", first_name="Alice")
    reply = _message(
        message_id=101,
        text="hello there",
        caption=None,
        from_user=reply_user,
        photo=None,
        document=None,
    )
    msg = _message(reply_to_message=reply)

    prefix, meta, key = TelegramChannel._extract_reply_context(msg)

    assert "hello there" in prefix
    assert "From: Alice (@alice)" in prefix
    assert "Message ID: 101" in prefix
    assert meta["reply_to_text"] == "hello there"
    assert meta["reply_to_message_id"] == 101
    assert meta["reply_to_user_id"] == 42
    assert key == "101"


def test_extract_reply_context_media_only():
    reply_user = _user(user_id=7, username=None, first_name="Bob")
    reply = _message(
        message_id=202,
        text=None,
        caption=None,
        from_user=reply_user,
        photo=[object()],
        document=SimpleNamespace(file_name="file.pdf"),
    )
    msg = _message(reply_to_message=reply)

    prefix, meta, key = TelegramChannel._extract_reply_context(msg)

    assert "Text: [none]" in prefix
    assert "Media: photo, document (file.pdf)" in prefix
    assert meta["reply_to_has_media"] is True
    assert meta["reply_to_media"] == "photo, document (file.pdf)"
    assert key == "202"
