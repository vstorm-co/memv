"""Tests for predict-calibrate extraction."""

from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

import pytest

from agent_memory.models import Episode, ExtractedKnowledge, Message, MessageRole, SemanticKnowledge
from agent_memory.processing.extraction import ExtractionResponse, PredictCalibrateExtractor

T = TypeVar("T")


class FakeLLMClient:
    """Fake LLM for testing extraction logic."""

    def __init__(self, prediction: str = "", extraction: list[ExtractedKnowledge] | None = None):
        self.prediction = prediction
        self.extraction = extraction or []
        self.generate_calls: list[str] = []
        self.generate_structured_calls: list[tuple[str, type]] = []

    async def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return self.prediction

    async def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        self.generate_structured_calls.append((prompt, response_model))
        return ExtractionResponse(extracted=self.extraction)  # type: ignore[return-value]


def make_message(role: MessageRole, content: str) -> Message:
    return Message(
        user_id="test-user",
        role=role,
        content=content,
        sent_at=datetime.now(timezone.utc),
    )


def make_episode(messages: list[Message]) -> Episode:
    return Episode(
        user_id="test-user",
        message_ids=[m.id for m in messages],
        title="Test Episode",
        narrative="A test episode.",
        start_time=messages[0].sent_at,
        end_time=messages[-1].sent_at,
    )


def make_knowledge(statement: str) -> SemanticKnowledge:
    return SemanticKnowledge(
        statement=statement,
        source_episode_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_extract_with_no_existing_knowledge():
    """When no existing knowledge, skip prediction and extract everything notable."""
    extracted = [
        ExtractedKnowledge(statement="User likes Python", knowledge_type="new", confidence=0.9),
        ExtractedKnowledge(statement="User works at Acme Corp", knowledge_type="new", confidence=0.8),
    ]
    llm = FakeLLMClient(extraction=extracted)
    extractor = PredictCalibrateExtractor(llm)

    messages = [
        make_message(MessageRole.USER, "I love Python and I work at Acme Corp"),
        make_message(MessageRole.ASSISTANT, "That's great!"),
    ]
    episode = make_episode(messages)

    result = await extractor.extract(episode, messages, existing_knowledge=[])

    # Should skip prediction (no generate call for prediction)
    assert len(llm.generate_calls) == 0
    # Should call generate_structured for extraction
    assert len(llm.generate_structured_calls) == 1
    assert result == extracted


@pytest.mark.asyncio
async def test_extract_with_existing_knowledge():
    """When existing knowledge present, predict first then extract gaps."""
    existing = [
        make_knowledge("User likes Python"),
        make_knowledge("User lives in Seattle"),
    ]
    extracted = [
        ExtractedKnowledge(statement="User changed jobs to Anthropic", knowledge_type="new", confidence=0.95),
    ]
    llm = FakeLLMClient(
        prediction="User probably discussed Python. Maybe mentioned Seattle.",
        extraction=extracted,
    )
    extractor = PredictCalibrateExtractor(llm)

    messages = [
        make_message(MessageRole.USER, "I just started at Anthropic!"),
        make_message(MessageRole.ASSISTANT, "Congratulations!"),
    ]
    episode = make_episode(messages)

    result = await extractor.extract(episode, messages, existing_knowledge=existing)

    # Should call generate for prediction
    assert len(llm.generate_calls) == 1
    assert "User likes Python" in llm.generate_calls[0]
    assert "User lives in Seattle" in llm.generate_calls[0]

    # Should call generate_structured for extraction
    assert len(llm.generate_structured_calls) == 1
    # The calibration prompt should include both prediction and raw content
    calibration_prompt = llm.generate_structured_calls[0][0]
    assert "probably discussed Python" in calibration_prompt
    assert "started at Anthropic" in calibration_prompt

    assert result == extracted


@pytest.mark.asyncio
async def test_extract_returns_empty_when_nothing_novel():
    """When everything is predicted, nothing should be extracted."""
    existing = [make_knowledge("User likes Python")]
    llm = FakeLLMClient(
        prediction="User probably mentioned they like Python.",
        extraction=[],  # Nothing novel
    )
    extractor = PredictCalibrateExtractor(llm)

    messages = [
        make_message(MessageRole.USER, "As you know, I really like Python"),
        make_message(MessageRole.ASSISTANT, "Yes, you've mentioned that!"),
    ]
    episode = make_episode(messages)

    result = await extractor.extract(episode, messages, existing_knowledge=existing)

    assert result == []


@pytest.mark.asyncio
async def test_format_messages():
    """Test message formatting for prompts."""
    extractor = PredictCalibrateExtractor(FakeLLMClient())

    messages = [
        make_message(MessageRole.USER, "Hello"),
        make_message(MessageRole.ASSISTANT, "Hi there"),
        make_message(MessageRole.USER, "How are you?"),
    ]

    formatted = extractor._format_messages(messages)

    assert "user: Hello" in formatted
    assert "assistant: Hi there" in formatted
    assert "user: How are you?" in formatted


@pytest.mark.asyncio
async def test_format_knowledge():
    """Test knowledge formatting for prompts."""
    extractor = PredictCalibrateExtractor(FakeLLMClient())

    knowledge = [
        make_knowledge("User likes Python"),
        make_knowledge("User works remotely"),
    ]

    formatted = extractor._format_knowledge(knowledge)

    assert "- User likes Python" in formatted
    assert "- User works remotely" in formatted
