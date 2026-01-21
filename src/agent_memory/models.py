from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from asyncio import Task


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(..., description="The ID of the user who sent the message.")
    role: MessageRole = Field(..., description="Indicates who sent the message.")
    content: str = Field(..., description="The actual message content.")
    sent_at: datetime = Field(..., description="When the message was sent (UTC).")


class Episode(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(..., description="The ID of user whose conversation this episode belongs to.")
    title: str = Field(..., description="The title of the episode.")
    content: str = Field(..., description="Detailed third-person narrative with ALL important information from the conversation.")
    original_messages: list[dict] = Field(..., description="Raw messages stored on the episode for extraction.")
    start_time: datetime = Field(..., description="Time when the episode started (UTC).")
    end_time: datetime = Field(..., description="Time when the episode ended (UTC).")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the episode was created (UTC).")

    @property
    def message_count(self) -> int:
        return len(self.original_messages)


class BiTemporalValidity(BaseModel):
    """
    Bi-temporal validity tracking.

    Event timeline (T): when the fact was/is true in the world
    Transaction timeline (T'): when we learned/recorded it
    """

    # Event time - when fact is true in world
    valid_at: datetime | None = Field(default=None, description="When fact became true (None = unknown/always)")
    invalid_at: datetime | None = Field(default=None, description="When fact stopped being true (None = still true)")

    # Transaction time - when we recorded it
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When we learned this")
    expired_at: datetime | None = Field(default=None, description="When we invalidated this record (None = current)")

    def is_valid_at(self, event_time: datetime) -> bool:
        """Check if fact was true at given event time."""
        if self.valid_at and event_time < self.valid_at:
            return False
        if self.invalid_at and event_time >= self.invalid_at:
            return False
        return True

    def is_current(self) -> bool:
        """Check if this is the current (non-expired) record."""
        return self.expired_at is None


class SemanticKnowledge(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    statement: str = Field(..., description="A declarative statement about the user or world generated from the conversation")
    source_episode_id: UUID = Field(..., description="The id of the episode that generated this knowledge")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="The time when the knowledge entry was created (UTC)."
    )
    importance_score: float | None = Field(default=None, description="The importance score of the knowledge entry.")
    embedding: list[float] | None = Field(default=None, description="The embedding of the statement.")

    # Bi-temporal validity fields
    valid_at: datetime | None = Field(default=None, description="When fact became true in world (None = unknown/always)")
    invalid_at: datetime | None = Field(default=None, description="When fact stopped being true (None = still true)")
    expired_at: datetime | None = Field(default=None, description="When this record was superseded (None = current)")

    def invalidate(self) -> None:
        """Mark this knowledge as superseded."""
        self.expired_at = datetime.now(timezone.utc)

    def is_valid_at(self, event_time: datetime) -> bool:
        """Check if fact was true at given event time."""
        if self.valid_at and event_time < self.valid_at:
            return False
        if self.invalid_at and event_time >= self.invalid_at:
            return False
        return True

    def is_current(self) -> bool:
        """Check if this is the current (non-expired) record."""
        return self.expired_at is None


class RetrievalResult(BaseModel):
    """Results from memory retrieval."""

    retrieved_knowledge: list[SemanticKnowledge] = Field(default_factory=list, description="Retrieved semantic knowledge entries.")
    retrieved_episodes: list["Episode"] = Field(default_factory=list, description="Retrieved episodes relevant to the query.")

    def as_text(self) -> str:
        """Simple text representation of knowledge statements."""
        return "\n".join(k.statement for k in self.retrieved_knowledge)

    def to_prompt(self) -> str:
        """
        Format retrieval results for LLM context injection.

        Groups knowledge by source episode and includes episode context
        to avoid redundancy.
        """
        lines = []

        # Build episode lookup for knowledge grouping
        episode_map = {ep.id: ep for ep in self.retrieved_episodes}
        knowledge_by_episode: dict[UUID, list[SemanticKnowledge]] = {}
        orphan_knowledge: list[SemanticKnowledge] = []

        for k in self.retrieved_knowledge:
            if k.source_episode_id in episode_map:
                knowledge_by_episode.setdefault(k.source_episode_id, []).append(k)
            else:
                orphan_knowledge.append(k)

        # Episodes with their knowledge
        if knowledge_by_episode:
            lines.append("## Relevant Context")
            for ep_id, knowledge_list in knowledge_by_episode.items():
                ep = episode_map[ep_id]
                lines.append(f"\n### {ep.title}")
                lines.append(f"_{ep.content}_")
                lines.append("\nKey facts:")
                for k in knowledge_list:
                    lines.append(f"- {k.statement}")

        # Episodes without extracted knowledge in results
        episodes_with_knowledge = set(knowledge_by_episode.keys())
        episodes_without = [ep for ep in self.retrieved_episodes if ep.id not in episodes_with_knowledge]
        if episodes_without:
            if not lines:
                lines.append("## Relevant Context")
            lines.append("\n### Related Conversations")
            for ep in episodes_without:
                lines.append(f"\n**{ep.title}**")
                lines.append(f"_{ep.content}_")

        # Orphan knowledge (episode not in results)
        if orphan_knowledge:
            lines.append("\n### Additional Facts")
            for k in orphan_knowledge:
                lines.append(f"- {k.statement}")

        return "\n".join(lines) if lines else "No relevant context found."


class ExtractedKnowledge(BaseModel):
    """Output of predict-calibrate extraction."""

    statement: str
    knowledge_type: Literal["new", "update", "contradiction"]
    temporal_info: str | None = None  # "since January 2024", "until next week"
    confidence: float = 1.0


class ProcessStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessTask(BaseModel):
    """Handle for monitoring/awaiting async processing."""

    model_config = {"arbitrary_types_allowed": True}

    user_id: str
    status: ProcessStatus = ProcessStatus.PENDING
    knowledge_count: int = 0
    error: str | None = None

    _task: Task[int] | None = PrivateAttr(default=None)

    @property
    def done(self) -> bool:
        """Check if processing has completed (success or failure)."""
        return self.status in (ProcessStatus.COMPLETED, ProcessStatus.FAILED)

    async def wait(self) -> int:
        """Wait for processing to complete and return knowledge count."""
        if self._task is None:
            return self.knowledge_count

        try:
            result = await self._task
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            # Status already updated by the task wrapper
            return 0
