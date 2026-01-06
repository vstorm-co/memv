from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
    message_ids: list[UUID] = Field(..., description="List of message ids of messages in the episode.")
    title: str = Field(..., description="The title of the episode.")
    narrative: str = Field(..., description="The narrative of the episode.")
    start_time: datetime = Field(..., description="Time when the episode started (UTC).")
    end_time: datetime = Field(..., description="Time when the episode ended (UTC).")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the episode was created (UTC).")


class SemanticKnowledge(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    statement: str = Field(..., description="A declarative statement about the user or world generated from the conversation")
    source_episode_id: UUID = Field(..., description="The id of the episode that generated this knowledge")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="The time when the knowledge entry was created (UTC)."
    )
    importance_score: float | None = Field(default=None, description="The importance score of the knowledge entry.")
    embedding: list[float] | None = Field(default=None, description="The embedding of the statement.")


class RetrievalResult(BaseModel):
    retrieved_knowledge: list[SemanticKnowledge] = Field(default_factory=list, description="The retrieved semantic knowledge entries.")

    def as_text(self) -> str:
        return "\n".join(k.statement for k in self.retrieved_knowledge)
