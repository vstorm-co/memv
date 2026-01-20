"""
Predict-calibrate extraction.

Nemori's core innovation: importance emerges from prediction error,
not upfront LLM scoring.
"""

from pydantic import BaseModel, Field

from agent_memory.models import Episode, ExtractedKnowledge, Message, SemanticKnowledge
from agent_memory.protocols import LLMClient


class ExtractionResponse(BaseModel):
    """Structured response from calibration LLM call."""

    extracted: list[ExtractedKnowledge] = Field(default_factory=list)


class PredictCalibrateExtractor:
    """
    Extract novel knowledge by comparing predictions against reality.

    Flow:
    1. Predict what the episode should contain (given existing knowledge)
    2. Compare prediction vs raw messages
    3. Extract only what we FAILED to predict
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def extract(
        self,
        episode: Episode,
        raw_messages: list[Message],
        existing_knowledge: list[SemanticKnowledge],
    ) -> list[ExtractedKnowledge]:
        """
        Core predict-calibrate loop.

        Args:
            episode: The episode to extract from
            raw_messages: Original messages (ground truth)
            existing_knowledge: Already-known facts to predict against

        Returns:
            Only genuinely novel knowledge
        """
        # Stage 1: Predict
        prediction = await self._predict(episode.title, existing_knowledge)

        # Stage 2: Calibrate
        raw_content = self._format_messages(raw_messages)
        novel = await self._extract_gaps(prediction, raw_content)

        return novel

    async def _predict(
        self,
        episode_title: str,
        existing_knowledge: list[SemanticKnowledge],
    ) -> str:
        """
        Predict what the episode likely contains based on existing KB.

        This is the "world model" generating expectations.
        """
        if not existing_knowledge:
            # No existing knowledge = can't predict anything
            # Everything in the episode is novel by definition
            return ""

        knowledge_text = self._format_knowledge(existing_knowledge)

        prompt = f"""You have the following knowledge about the user:

<existing_knowledge>
{knowledge_text}
</existing_knowledge>

A conversation episode titled "{episode_title}" just occurred.

Based on what you already know, predict what information, facts, or events this episode likely contains. Be specific about:
- Topics the user probably discussed
- Preferences or opinions they might have expressed
- Decisions or plans they may have mentioned
- Any updates to known facts

Write your prediction as a list of expected statements."""

        return await self.llm.generate(prompt)

    async def _extract_gaps(
        self,
        prediction: str,
        raw_content: str,
    ) -> list[ExtractedKnowledge]:
        """
        Compare prediction to reality, extract what we missed.

        Importance emerges here:
        - Predicted correctly → already known → don't extract
        - Prediction failed → novel information → extract
        """
        if not prediction:
            # No prediction = extract everything notable
            prompt = f"""Extract important knowledge from this conversation.

<conversation>
{raw_content}
</conversation>

Extract facts that would be useful to remember about the user, including:
- Preferences, opinions, likes/dislikes
- Facts about their life, work, relationships
- Decisions, plans, or intentions
- Any information with lasting relevance

Do NOT extract:
- Generic pleasantries or filler
- Temporary/ephemeral information
- Things that are obvious or common knowledge

For each item, specify:
- statement: A declarative sentence about what you learned
- knowledge_type: "new" (since we have no prior knowledge)
- temporal_info: When this became true, if mentioned (e.g., "since January 2024", "starting next month")
- confidence: 0.0-1.0 how certain you are this is accurate"""

        else:
            prompt = f"""Compare the predicted episode content against what actually happened.

<prediction>
{prediction}
</prediction>

<actual_conversation>
{raw_content}
</actual_conversation>

Extract ONLY information that:
1. Was NOT correctly predicted (novel/surprising)
2. Updates or contradicts existing knowledge
3. Contains new facts, preferences, or relationships

Do NOT extract:
- Information that was correctly anticipated
- Generic/obvious statements
- Information without lasting relevance

For each extracted item, specify:
- statement: A declarative sentence about what you learned
- knowledge_type: "new" if entirely new, "update" if it refines existing knowledge, "contradiction" if it conflicts with what we knew
- temporal_info: When this became true, if mentioned (e.g., "since January 2024", "starting next month")
- confidence: 0.0-1.0 how certain you are this is accurate"""

        response = await self.llm.generate_structured(prompt, ExtractionResponse)
        return response.extracted

    def _format_messages(self, messages: list[Message]) -> str:
        """Format raw messages for the prompt."""
        lines = []
        for msg in messages:
            lines.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(lines)

    def _format_knowledge(self, knowledge: list[SemanticKnowledge]) -> str:
        """Format existing knowledge for the prompt."""
        return "\n".join(f"- {k.statement}" for k in knowledge)
