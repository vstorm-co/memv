"""
Episode boundary detection.

Detects semantic boundaries in message streams to segment
conversations into coherent episodes.
"""

from pydantic import BaseModel

from agent_memory.models import Message
from agent_memory.protocols import LLMClient


class BoundarySignal(BaseModel):
    """Output of boundary detection."""

    is_boundary: bool
    confidence: float  # 0.0 - 1.0
    reason: str | None = None


class BoundaryDetector:
    """
    Detects episode boundaries in message streams.

    Uses LLM to evaluate semantic shifts based on:
    - Contextual coherence (is new message related to current topic?)
    - Temporal markers ("by the way", "anyway", "on another note")
    - Intent shifts (info-seeking → decision-making)

    Boundary triggers when:
    - (is_boundary AND confidence > threshold) OR
    - (buffer_size >= max_messages)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        confidence_threshold: float = 0.7,
        max_buffer_size: int = 25,
    ):
        self.llm = llm_client
        self.confidence_threshold = confidence_threshold
        self.max_buffer_size = max_buffer_size

    async def should_segment(
        self,
        new_message: Message,
        buffer: list[Message],
    ) -> tuple[bool, BoundarySignal | None]:
        """
        Determine if we should start a new episode.

        Args:
            new_message: The incoming message
            buffer: Current episode's messages

        Returns:
            (should_segment, signal) - signal is None if hard limit triggered
        """
        # Hard limit: always segment if buffer full
        if len(buffer) >= self.max_buffer_size:
            return True, None

        # Empty buffer: don't segment, just add
        if not buffer:
            return False, None

        # LLM-based boundary detection
        signal = await self._detect_boundary(new_message, buffer)

        should_segment = signal.is_boundary and signal.confidence >= self.confidence_threshold

        return should_segment, signal

    async def _detect_boundary(
        self,
        new_message: Message,
        buffer: list[Message],
    ) -> BoundarySignal:
        """Use LLM to detect if new message crosses a topic boundary."""

        # Format conversation context
        context = self._format_context(buffer)
        new_content = f"{new_message.role.value}: {new_message.content}"

        prompt = f"""Analyze whether the new message represents a topic or intent shift from the current conversation.

<current_conversation>
{context}
</current_conversation>

<new_message>
{new_content}
</new_message>

Evaluate these signals:
1. **Topic shift**: Does the new message introduce a completely different subject?
2. **Temporal markers**: Does it contain phrases like "by the way", "anyway", "on another note", "speaking of something else"?
3. **Intent shift**: Does the user's goal change (e.g., from asking questions to making decisions, from one task to another)?
4. **Contextual coherence**: How related is the new message to the ongoing discussion?

Respond with JSON:
{{
    "is_boundary": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}

A boundary should be detected when there's a clear topic change, NOT for:
- Follow-up questions on the same topic
- Clarifications or elaborations
- Natural conversation flow within the same subject
"""

        response = await self.llm.generate(prompt)

        # Parse response
        return self._parse_response(response)

    def _format_context(self, messages: list[Message], max_messages: int = 10) -> str:
        """Format recent messages for context."""
        # Take last N messages to keep prompt manageable
        recent = messages[-max_messages:] if len(messages) > max_messages else messages

        lines = []
        for msg in recent:
            lines.append(f"{msg.role.value}: {msg.content}")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> BoundarySignal:
        """Parse LLM response into BoundarySignal."""
        import json

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                response = response[start:end]

            data = json.loads(response.strip())

            return BoundarySignal(
                is_boundary=bool(data.get("is_boundary", False)),
                confidence=float(data.get("confidence", 0.0)),
                reason=data.get("reason"),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # Default to no boundary on parse failure
            return BoundarySignal(
                is_boundary=False,
                confidence=0.0,
                reason="Failed to parse LLM response",
            )
