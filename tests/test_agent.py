"""BugTraceAgent LangGraph 单元测试。"""

from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.agent.graph import (
    create_agent_graph,
)
from app.agent.nodes import (
    should_continue,
)


@pytest.fixture
def mock_mcp_manager():
    """创建假的 MCP Manager。"""

    manager = MagicMock()

    manager.call_tool = AsyncMock(return_value={"content": "mock github result"})

    return manager


class TestAgentGraph:
    """测试 BugTraceAgent Graph。"""

    def test_create_agent_graph(
        self,
        mock_mcp_manager,
    ):
        """提供 MCP Manager 后应该能正常创建 Graph。"""

        graph = create_agent_graph(mcp_manager=mock_mcp_manager)

        assert graph is not None

    def test_create_agent_graph_without_mcp(self):
        """没有 GitHub MCP 时应该拒绝创建 Agent。"""

        with pytest.raises(
            RuntimeError,
            match="GitHub MCP",
        ):
            create_agent_graph()

    @pytest.mark.asyncio
    async def test_agent_graph_execution(
        self,
        mock_mcp_manager,
    ):
        """
        Mock DeepSeek，
        验证 Agent 可以完成一次无 Tool Call 对话。
        """

        mock_response = AIMessage(
            content="这是模拟的中文分析结果。",
            tool_calls=[],
        )

        with patch("app.agent.nodes.get_llm") as mock_get_llm:

            mock_bound_llm = MagicMock()

            mock_bound_llm.ainvoke = AsyncMock(return_value=mock_response)

            mock_llm = MagicMock()

            mock_llm.bind_tools.return_value = mock_bound_llm

            mock_get_llm.return_value = mock_llm

            graph = create_agent_graph(mcp_manager=mock_mcp_manager)

            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content="分析测试仓库")],
                    "session_id": ("unit-test-session"),
                }
            )

            assert "messages" in result

            assert result["messages"][-1].content == "这是模拟的中文分析结果。"


class TestAgentRouting:
    """测试 LangGraph 路由判断。"""

    def test_continue_when_tool_call_exists(
        self,
    ):
        """LLM 请求调用工具时应该进入 tools 节点。"""

        mock_message = MagicMock()

        mock_message.tool_calls = [
            {
                "name": "github_search_code",
                "args": {
                    "owner": "Abby-statistic",
                    "repo": "CAMPUS",
                    "query": "routing_score",
                },
            }
        ]

        state = {"messages": [mock_message]}

        assert should_continue(state) == "continue"

    def test_end_without_tool_calls(
        self,
    ):
        """没有 Tool Call 时应该结束 Agent。"""

        mock_message = MagicMock()

        mock_message.tool_calls = []

        state = {"messages": [mock_message]}

        assert should_continue(state) == "end"
