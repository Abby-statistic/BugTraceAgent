"""Pydantic schemas for API request and response models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")


class ChatRequest(BaseModel):
    """Chat request with user message."""
    
    message: str = Field(..., description="User message to send to the agent", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")


class ChatResponse(BaseModel):
    """Chat response from the agent."""
    
    response: str = Field(..., description="Agent's response message")
    session_id: str = Field(..., description="Session ID for this conversation")
    tool_calls: list[dict] | None = Field(
        default=None,
        description="List of tools called during processing (for debugging)",
    )


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")


class StreamEventType(str, Enum):
    """Types of events in SSE stream."""
    
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """Event in SSE stream."""
    
    type: StreamEventType = Field(..., description="Event type")
    data: dict = Field(..., description="Event data")
