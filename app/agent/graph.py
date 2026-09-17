"""BugTraceAgent LangGraph 工作流。"""

from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    build_github_tools,
    create_call_model,
    create_tool_node,
    should_continue,
)
from app.agent.state import AgentState
from app.mcp.client import MCPClientManager


def create_agent_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    mcp_manager: MCPClientManager | None = None,
):
    """创建 BugTraceAgent 工作流。"""

    if mcp_manager is None:
        raise RuntimeError("GitHub MCP Server 尚未连接")

    tools = build_github_tools(mcp_manager)

    call_model = create_call_model(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "agent",
        call_model,
    )

    workflow.add_node(
        "tools",
        create_tool_node(tools),
    )

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    workflow.add_edge(
        "tools",
        "agent",
    )

    return workflow.compile(checkpointer=checkpointer)
