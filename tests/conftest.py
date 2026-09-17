"""Pytest 公共 fixtures。"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """创建测试客户端，并 Mock 外部依赖。"""

    # 测试环境不真正连接 PostgreSQL / GitHub MCP
    app.state.checkpointer = MagicMock(name="mock_checkpointer")

    app.state.mcp_manager = MagicMock(name="mock_mcp_manager")

    with TestClient(app) as test_client:
        # lifespan 有可能覆盖 state，所以再设置一次
        test_client.app.state.checkpointer = MagicMock(name="mock_checkpointer")

        test_client.app.state.mcp_manager = MagicMock(name="mock_mcp_manager")

        yield test_client
