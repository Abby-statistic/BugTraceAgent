"""Tests for API endpoints."""

from unittest.mock import AsyncMock, patch


def test_health_check(client):
    """测试健康检查接口。"""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_chat_endpoint_success(client):
    """测试成功的聊天请求。"""

    mock_result = {
        "messages": [
            type(
                "MockMessage",
                (),
                {
                    "content": "这是模拟的中文分析结果。",
                    "tool_calls": [],
                },
            )()
        ]
    }

    with patch("app.api.routes.create_agent_graph") as mock_graph:

        mock_graph.return_value.ainvoke = AsyncMock(return_value=mock_result)

        response = client.post(
            "/chat",
            json={"message": "请分析测试仓库"},
        )

        assert response.status_code == 200

        data = response.json()

        assert data["response"] == "这是模拟的中文分析结果。"


def test_chat_endpoint_validation(client):
    """测试空消息校验。"""

    response = client.post(
        "/chat",
        json={"message": ""},
    )

    assert response.status_code == 422


def test_chat_endpoint_missing_message(client):
    """测试缺少 message 字段。"""

    response = client.post(
        "/chat",
        json={},
    )

    assert response.status_code == 422
