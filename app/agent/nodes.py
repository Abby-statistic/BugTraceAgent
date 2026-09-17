"""BugTraceAgent 的 LangGraph 节点。

负责：
1. 将 GitHub MCP 能力包装成 LangChain Tool
2. 调用 DeepSeek
3. 判断是否继续执行工具
"""

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.core.llm_factory import get_llm
from app.mcp.client import MCPClientManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
你是 BugTraceAgent，一个基于 GitHub 真实仓库证据进行代码检索、
Issue 分析和软件问题根因定位的智能体。

你必须始终使用中文回答。

## 核心原则

1. 只要问题涉及具体 GitHub 仓库的文件、代码、Issue、函数、
   类、变量、架构或实现逻辑，必须先调用 GitHub 工具取证，
   不允许仅凭模型记忆回答。

2. 不得编造不存在的文件、目录、函数、类、变量、Issue、
   代码逻辑或调用关系。

3. `github_search_code` 主要用于定位候选文件。
   在分析某个文件的具体实现之前，应继续使用
   `github_read_path` 读取真实文件内容。

4. 没有实际读取过的代码，不得声称已经分析过。

5. 必须区分：
   - 已确认事实：直接来自 GitHub 工具返回结果；
   - 分析推断：基于真实证据形成的判断；
   - 根因假设：当前最可能但尚未完全验证的原因。

6. 如果证据不足，应继续调用工具获取信息；
   如果仍无法确认，明确说明“目前证据不足，无法确认”，
   不要为了完整性而猜测。

## 推荐分析流程

对于 Bug / Issue / 根因分析：

用户问题
→ 如果有 Issue 编号，先读取 Issue
→ 查看仓库结构（必要时）
→ 搜索相关代码
→ 读取最相关的真实文件
→ 必要时搜索历史 Issue
→ 汇总证据
→ 形成根因假设
→ 给出修改与验证建议

避免无意义地调用过多工具。
简单目录或文件查询只执行完成任务所需的工具。

## 回答要求

如果用户只是查询目录、文件、代码位置或普通代码逻辑，
直接使用自然、简洁的中文回答，不必套用完整分析模板。

如果用户要求 Bug、Issue、故障、根因或工程风险分析，
使用以下格式：

## 问题类型

## 问题摘要

## 疑似相关文件

只列出已经通过 GitHub 工具确认存在的文件。

## 代码证据

说明实际读取或搜索到了什么，以及证据与问题的关系。

## 根因假设

明确区分事实和推测。
如果存在多个合理原因，可以分别列出。
证据不足时明确说明。

## 修改建议

尽量指出：
- 修改位置
- 修改目标
- 原因
- 潜在影响
- 修改后的验证方法

## 置信度

使用 0%～100%。

置信度应与证据强度一致：
- 90% 以上：关键 Issue 和代码均已读取，证据链完整；
- 75%～89%：代码证据较强，但缺少部分运行时验证；
- 50%～74%：存在合理线索，但仍有多个解释；
- 50% 以下：缺少关键代码或上下文。

普通目录和文件查询无需提供置信度。

## 安全与权限

当前 Agent 只负责分析，不修改 GitHub 仓库。
不得声称已经创建 Commit、Issue、Pull Request 或修改代码。

工具调用失败时，不得伪造结果，应说明哪些信息因此无法确认。

不要输出内部思维过程。
只提供关键证据、分析结论、必要解释和修改建议。
"""


def _mcp_result_to_text(
    result: Any,
) -> str:
    """把 MCP 返回值转换为文本。"""

    if hasattr(
        result,
        "model_dump_json",
    ):
        return result.model_dump_json()

    if hasattr(
        result,
        "model_dump",
    ):
        return json.dumps(
            result.model_dump(),
            ensure_ascii=False,
            default=str,
        )

    return str(result)


def build_github_tools(
    mcp_manager: MCPClientManager,
) -> list:
    """创建给 DeepSeek 使用的 GitHub 工具。"""

    @tool
    async def github_read_issue(
        owner: str,
        repo: str,
        issue_number: int,
    ) -> str:
        """
        读取指定 GitHub Issue 的详细内容。

        当用户提供 Issue 编号时优先使用。

        参数：
        owner：GitHub 用户名或组织名。
        repo：仓库名称。
        issue_number：Issue 编号。
        """

        result = await mcp_manager.call_tool(
            "github",
            "issue_read",
            {
                "method": "get",
                "owner": owner,
                "repo": repo,
                "issue_number": issue_number,
            },
        )

        return _mcp_result_to_text(result)

    @tool
    async def github_read_path(
        owner: str,
        repo: str,
        path: str = "",
    ) -> str:
        """
        读取 GitHub 仓库中的文件或者目录。

        path 为空字符串时读取仓库根目录。

        参数：
        owner：GitHub 用户名或组织名。
        repo：仓库名称。
        path：文件或目录路径。
        """

        result = await mcp_manager.call_tool(
            "github",
            "get_file_contents",
            {
                "owner": owner,
                "repo": repo,
                "path": path,
            },
        )

        return _mcp_result_to_text(result)

    @tool
    async def github_search_code(
        owner: str,
        repo: str,
        query: str,
    ) -> str:
        """
        在指定 GitHub 仓库中搜索源代码。

        用于根据函数名、类名、关键词、
        错误信息等定位相关代码。
        """

        scoped_query = f"{query} repo:{owner}/{repo}"

        result = await mcp_manager.call_tool(
            "github",
            "search_code",
            {
                "query": scoped_query,
                "perPage": 10,
            },
        )

        return _mcp_result_to_text(result)

    @tool
    async def github_search_issues(
        owner: str,
        repo: str,
        query: str,
    ) -> str:
        """
        搜索指定仓库中的历史 Issue。

        用于寻找类似 Bug、历史讨论或者
        已经出现过的问题。
        """

        result = await mcp_manager.call_tool(
            "github",
            "search_issues",
            {
                "owner": owner,
                "repo": repo,
                "query": query,
                "perPage": 10,
            },
        )

        return _mcp_result_to_text(result)

    return [
        github_read_issue,
        github_read_path,
        github_search_code,
        github_search_issues,
    ]


def create_tool_node(
    tools: list,
) -> ToolNode:
    """创建 LangGraph 工具执行节点。"""

    return ToolNode(tools)


def create_call_model(
    tools: list,
):
    """创建调用 DeepSeek 的 Agent 节点。"""

    async def call_model(
        state: AgentState,
    ) -> dict:

        try:

            llm = get_llm()

            llm_with_tools = llm.bind_tools(tools)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                *state["messages"],
            ]

            logger.info(
                "调用 DeepSeek，消息数量：%d",
                len(messages),
            )

            response = await llm_with_tools.ainvoke(messages)

            logger.info(
                "DeepSeek Tool Calls：%s",
                getattr(
                    response,
                    "tool_calls",
                    None,
                ),
            )

            if not response.content and not getattr(
                response,
                "tool_calls",
                None,
            ):
                return {"messages": [AIMessage(content=("模型没有返回有效结果，" "请重新尝试。"))]}

            return {"messages": [response]}

        except Exception as exc:

            logger.exception("调用 DeepSeek 失败")

            return {"messages": [AIMessage(content=("分析过程中发生错误：" f"{exc}"))]}

    return call_model


def should_continue(
    state: AgentState,
) -> str:
    """判断下一步是否执行工具。"""

    last_message = state["messages"][-1]

    if (
        hasattr(
            last_message,
            "tool_calls",
        )
        and last_message.tool_calls
    ):
        return "continue"

    return "end"
