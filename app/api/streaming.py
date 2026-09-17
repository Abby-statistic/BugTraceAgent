"""Streaming utilities for Server-Sent Events (SSE).

Provides async generators for streaming agent responses in real-time.
"""

import json
from typing import AsyncIterator

from langchain_core.messages import AIMessage, ToolMessage


class StreamEventType:
    """Event types for SSE streaming."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"
    ERROR = "error"
    DONE = "done"


async def stream_agent_response(
    graph,
    input_data: dict,
    config: dict | None = None,
) -> AsyncIterator[str]:
    """
    Stream agent execution events via Server-Sent Events.

    Args:
        graph: Compiled LangGraph workflow
        input_data: Input data for the agent

    Yields:
        SSE-formatted event strings
    """
    try:
        # Stream events from the graph
        async for event in graph.astream(
            input_data,
            config=config,
        ):
            # Handle different event types
            for node_name, node_output in event.items():
                if node_name == "agent":
                    # Agent thinking/decision
                    messages = node_output.get("messages", [])
                    if messages:
                        last_message = messages[-1]

                        # Check if it's an AI message with tool calls
                        if isinstance(last_message, AIMessage):
                            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                                # Agent decided to call tools
                                for tool_call in last_message.tool_calls:
                                    yield format_sse_event(
                                        StreamEventType.TOOL_CALL,
                                        {
                                            "name": tool_call.get("name"),
                                            "args": tool_call.get("args"),
                                        },
                                    )
                            elif last_message.content:
                                # Agent has final answer
                                yield format_sse_event(
                                    StreamEventType.ANSWER, {"content": last_message.content}
                                )

                elif node_name == "tools":
                    # Tool execution results
                    messages = node_output.get("messages", [])
                    if messages:
                        for message in messages:
                            if isinstance(message, ToolMessage):
                                yield format_sse_event(
                                    StreamEventType.TOOL_RESULT,
                                    {
                                        "tool": (
                                            message.name if hasattr(message, "name") else "unknown"
                                        ),
                                        "result": message.content,
                                    },
                                )

        # Send done event
        yield format_sse_event(StreamEventType.DONE, {})

    except Exception as e:
        # Send error event
        yield format_sse_event(StreamEventType.ERROR, {"message": str(e)})


def format_sse_event(event_type: str, data: dict) -> str:
    """
    Format data as SSE event.

    Args:
        event_type: Type of event
        data: Event data

    Returns:
        SSE-formatted string
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
