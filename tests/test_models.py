from datetime import datetime, timedelta, timezone

from agentmemory.models import Episode, Message, MessageRole, RetrievalResult, SemanticKnowledge


def test_message_roundtrip():
    msg = Message(
        user_id="user123",
        role=MessageRole.USER,
        content="Hello",
        sent_at=datetime.now(timezone.utc),
    )
    data = msg.model_dump()
    restored = Message.model_validate(data)
    assert restored == msg


def test_episode_roundtrip():
    messages = [
        Message(
            user_id="user123",
            role=MessageRole.USER,
            content="Hello",
            sent_at=datetime.now(timezone.utc),
        ),
        Message(
            user_id="user123",
            role=MessageRole.ASSISTANT,
            content="Hi",
            sent_at=datetime.now(timezone.utc),
        ),
    ]

    episode = Episode(
        title="Test Episode",
        narrative="Testing episode",
        user_id="user123",
        message_ids=[message.id for message in messages],
        start_time=datetime.now(timezone.utc) - timedelta(seconds=13),
        end_time=datetime.now(timezone.utc) - timedelta(seconds=6),
    )
    data = episode.model_dump()
    restored = Episode.model_validate(data)
    assert restored == episode
