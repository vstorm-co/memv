"""
Episode generation.

Transforms segmented conversations into structured episodic memories
with titles and third-person narratives.
"""

from datetime import datetime

from agent_memory.models import Episode, Message
from agent_memory.protocols import LLMClient


class EpisodeGenerator:
    """
    Generates episodic memories from conversation segments.

    Transforms raw message sequences into structured episodes with:
    - Title: Concise summary of the episode's core theme
    - Narrative: Third-person account preserving key information

    This implements Nemori's "Representation Alignment" principle.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate(
        self,
        messages: list[Message],
        user_id: str,
    ) -> Episode:
        """
        Generate an episode from a conversation segment.

        Args:
            messages: The messages comprising this episode
            user_id: User ID for the episode

        Returns:
            A structured Episode with title and narrative
        """
        if not messages:
            raise ValueError("Cannot generate episode from empty message list")

        # Format conversation for LLM
        conversation = self._format_conversation(messages)

        # Get reference timestamp for resolving relative dates
        reference_time = messages[-1].sent_at

        # Generate title and narrative
        title, narrative = await self._generate_title_and_narrative(conversation, reference_time)

        # Build episode
        return Episode(
            user_id=user_id,
            message_ids=[m.id for m in messages],
            title=title,
            narrative=narrative,
            start_time=messages[0].sent_at,
            end_time=messages[-1].sent_at,
        )

    async def _generate_title_and_narrative(
        self,
        conversation: str,
        reference_time: datetime,
    ) -> tuple[str, str]:
        """Generate title and narrative via LLM."""

        prompt = f"""Transform this conversation into a structured episodic memory.

<conversation>
{conversation}
</conversation>

<reference_timestamp>
{reference_time.isoformat()}
</reference_timestamp>

Generate:
1. **Title**: A concise phrase (3-7 words) capturing the episode's main theme
2. **Narrative**: A brief third-person summary (1-3 sentences max) that:
   - States only the key facts and outcomes
   - Converts relative dates to absolute dates using the reference timestamp
   - Omits pleasantries, filler, and obvious context

Keep it concise. The narrative should be useful context, not a retelling.

Respond with JSON:
{{
    "title": "...",
    "narrative": "..."
}}
"""

        response = await self.llm.generate(prompt)
        return self._parse_response(response)

    def _format_conversation(self, messages: list[Message]) -> str:
        """Format messages into readable conversation."""
        lines = []
        for msg in messages:
            timestamp = msg.sent_at.strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{timestamp}] {msg.role.value}: {msg.content}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response into title and narrative."""
        import json

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

            title = data.get("title", "Untitled Episode")
            narrative = data.get("narrative", "")

            if not narrative:
                raise ValueError("Empty narrative")

            return title, narrative

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: use raw response as narrative
            return "Conversation Episode", response.strip()
