"""Tests for streaming endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_stream_endpoint_success(client):
    """Test streaming endpoint with successful response."""
    # Mock the agent graph
    mock_event_1 = {
        "agent": {
            "messages": [
                MagicMock(
                    content="",
                    tool_calls=[
                        {
                            "name": "github_search_code",
                            "args": {
                                "owner": "Abby-statistic",
                                "repo": "CAMPUS",
                                "query": "routing_score",
                            },
                        }
                    ],
                )
            ]
        }
    }

    mock_event_2 = {
        "tools": {
            "messages": [
                MagicMock(
                    name="github_search_code",
                    content='{"items": [{"path": "CampusAssistant/example.py"}]}',
                )
            ]
        }
    }

    mock_event_3 = {
        "agent": {"messages": [MagicMock(content="I've added the task!", tool_calls=[])]}
    }

    async def mock_astream(input_data):
        yield mock_event_1
        yield mock_event_2
        yield mock_event_3

    with patch("app.api.routes.create_agent_graph") as mock_graph:
        mock_graph_instance = MagicMock()
        mock_graph_instance.astream = mock_astream
        mock_graph.return_value = mock_graph_instance

        response = client.get(
            "/chat/stream",
            params={"message": "请搜索 CAMPUS 中 routing_score 的实现"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


def test_stream_endpoint_missing_message(client):
    """Test streaming endpoint with missing message parameter."""
    response = client.get("/chat/stream")

    assert response.status_code == 422  # Validation error


def test_stream_endpoint_empty_message(client):
    """Test streaming endpoint with empty message."""
    response = client.get("/chat/stream?message=")

    assert response.status_code == 422  # Validation error
