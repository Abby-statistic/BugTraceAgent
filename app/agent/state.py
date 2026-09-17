"""Agent state schema for LangGraph workflow."""

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State schema for the agent graph."""
    
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str  # For conversation memory
