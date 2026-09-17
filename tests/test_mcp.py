"""GitHub MCP 工具单元测试。

这里不真正访问 GitHub。
通过 Mock MCPClientManager 验证：
1. Agent 暴露了正确的 GitHub 工具
2. 工具调用被正确转发给 MCP Server
3. 参数构造正确
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.nodes import build_github_tools


@pytest.fixture
def mock_mcp_manager():
    """创建假的 MCP Manager，避免测试真正访问 GitHub。"""

    manager = MagicMock()

    manager.call_tool = AsyncMock(return_value={"content": "mock github result"})

    return manager


@pytest.fixture
def github_tools(mock_mcp_manager):
    """创建 BugTraceAgent 的 GitHub 工具。"""

    tools = build_github_tools(mock_mcp_manager)

    return {tool.name: tool for tool in tools}


def test_github_tools_are_registered(
    github_tools,
):
    """应该注册当前 BugTraceAgent 需要的 4 个工具。"""

    assert set(github_tools.keys()) == {
        "github_read_issue",
        "github_read_path",
        "github_search_code",
        "github_search_issues",
    }


@pytest.mark.asyncio
async def test_github_read_issue(
    github_tools,
    mock_mcp_manager,
):
    """读取 Issue 时应该调用 GitHub MCP 的 issue_read。"""

    tool = github_tools["github_read_issue"]

    await tool.ainvoke(
        {
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "issue_number": 1,
        }
    )

    mock_mcp_manager.call_tool.assert_awaited_once_with(
        "github",
        "issue_read",
        {
            "method": "get",
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "issue_number": 1,
        },
    )


@pytest.mark.asyncio
async def test_github_read_repository_root(
    github_tools,
    mock_mcp_manager,
):
    """读取仓库根目录时应该调用 get_file_contents。"""

    tool = github_tools["github_read_path"]

    await tool.ainvoke(
        {
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "path": "",
        }
    )

    mock_mcp_manager.call_tool.assert_awaited_once_with(
        "github",
        "get_file_contents",
        {
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "path": "",
        },
    )


@pytest.mark.asyncio
async def test_github_search_code(
    github_tools,
    mock_mcp_manager,
):
    """代码搜索应该自动限制到目标仓库。"""

    tool = github_tools["github_search_code"]

    await tool.ainvoke(
        {
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "query": "routing_score",
        }
    )

    mock_mcp_manager.call_tool.assert_awaited_once_with(
        "github",
        "search_code",
        {
            "query": ("routing_score " "repo:Abby-statistic/CAMPUS"),
            "perPage": 10,
        },
    )


@pytest.mark.asyncio
async def test_github_search_issues(
    github_tools,
    mock_mcp_manager,
):
    """历史 Issue 搜索应该正确转发给 GitHub MCP。"""

    tool = github_tools["github_search_issues"]

    await tool.ainvoke(
        {
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "query": "timeout",
        }
    )

    mock_mcp_manager.call_tool.assert_awaited_once_with(
        "github",
        "search_issues",
        {
            "owner": "Abby-statistic",
            "repo": "CAMPUS",
            "query": "timeout",
            "perPage": 10,
        },
    )
