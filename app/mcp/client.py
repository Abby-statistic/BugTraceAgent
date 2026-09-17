"""MCP 客户端管理器。

负责读取 mcp_servers.json，
连接 GitHub MCP Server，
并管理 MCP Session 生命周期。
"""

import asyncio
import json
import logging
import os

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# 将 .env 中的 GitHub Token 加载到 os.environ，
# 这样 Docker 子进程才能读取。
load_dotenv()


class MCPServerConfig:
    """单个 MCP Server 的配置。"""

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
    ):
        self.name = name
        self.type = config.get("type", "stdio")
        self.enabled = config.get("enabled", True)
        self.description = config.get("description", "")
        self.command = config.get("command")
        self.args = config.get("args", [])
        self.env = config.get("env", {})


class MCPClientManager:
    """统一管理 MCP Server 连接。"""

    def __init__(
        self,
        config_path: str = "mcp_servers.json",
    ):
        self.config_path = config_path

        self.servers: dict[str, MCPServerConfig] = {}
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, list[Any]] = {}

        self._exit_stack: AsyncExitStack | None = None

        self._load_config()

    def _load_config(self) -> None:
        """读取 mcp_servers.json。"""

        config_file = Path(self.config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"MCP 配置文件不存在：{self.config_path}")

        with open(
            config_file,
            "r",
            encoding="utf-8",
        ) as f:
            config_data = json.load(f)

        for name, server_config in config_data.get("servers", {}).items():
            self.servers[name] = MCPServerConfig(
                name,
                server_config,
            )

            logger.info(
                "已加载 MCP Server 配置：%s",
                name,
            )

    async def __aenter__(self):
        """启动 MCP Manager。"""

        self._exit_stack = AsyncExitStack()

        await self._exit_stack.__aenter__()

        await self.connect_all()

        return self

    async def __aexit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        """关闭 MCP Manager。"""

        await self.disconnect_all()

    async def connect_all(self) -> None:
        """连接所有已启用的 MCP Server。"""

        for name, config in self.servers.items():

            if not config.enabled:
                logger.info(
                    "跳过未启用 MCP Server：%s",
                    name,
                )
                continue

            if config.type == "builtin":
                logger.info(
                    "跳过 builtin Server：%s",
                    name,
                )
                continue

            await self._connect_server(
                name,
                config,
            )

    async def _connect_server(
        self,
        name: str,
        config: MCPServerConfig,
    ) -> None:
        """连接单个 stdio MCP Server。"""

        if self._exit_stack is None:
            raise RuntimeError("MCPClientManager 尚未进入 async context")

        env = os.environ.copy()

        for key, value in config.env.items():

            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_name = value[2:-1]

                resolved_value = os.environ.get(
                    env_name,
                    "",
                )

                if not resolved_value:
                    raise RuntimeError(f"环境变量 {env_name} 没有配置")

                env[key] = resolved_value

            else:
                env[key] = str(value)

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env,
        )

        logger.info(
            "正在连接 MCP Server：%s",
            name,
        )

        async with asyncio.timeout(30):

            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            session = ClientSession(
                read_stream,
                write_stream,
            )

            session = await self._exit_stack.enter_async_context(session)

            await session.initialize()

            tools_result = await session.list_tools()

            self.sessions[name] = session

            self.tools[name] = list(tools_result.tools)

        logger.info(
            "MCP Server %s 已连接，共加载 %d 个工具",
            name,
            len(self.tools[name]),
        )

    async def disconnect_all(self) -> None:
        """关闭所有 MCP 连接。"""

        self.sessions.clear()
        self.tools.clear()

        if self._exit_stack is not None:

            await self._exit_stack.aclose()

            self._exit_stack = None

        logger.info("所有 MCP Server 已关闭")

    def get_all_tools(
        self,
    ) -> dict[str, list[Any]]:
        """返回 MCP Server 暴露的工具。"""

        return self.tools.copy()

    def get_enabled_servers(
        self,
    ) -> list[str]:
        """返回已启用 Server 名称。"""

        return [name for name, config in self.servers.items() if config.enabled]

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """调用指定 MCP Tool。"""

        if server_name not in self.sessions:
            raise RuntimeError(f"MCP Server 未连接：{server_name}")

        session = self.sessions[server_name]

        logger.info(
            "调用 MCP Tool：%s.%s",
            server_name,
            tool_name,
        )

        return await session.call_tool(
            tool_name,
            arguments,
        )
