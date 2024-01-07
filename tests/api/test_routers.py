import pytest
from fastapi import WebSocketDisconnect
from src.api.routers import websocket_endpoint
from unittest.mock import AsyncMock

from unittest.mock import patch


@pytest.fixture
@pytest.mark.asyncio
@patch("src.api.routers.OpenAIBackend")
@patch("src.api.routers.StreamingLLMCallbackHandler")
@patch("src.api.routers.manager.connect")
async def test_websocket_endpoint_openai(
    monkeypatch, mock_connect, mock_handler, mock_openai
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key-test")
    monkeypatch.setenv("ENDPOINT_TYPE", "openai")
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.return_value = "Hello, world!"
    await websocket_endpoint(mock_websocket)
    mock_websocket.send_json.assert_called()
    mock_connect.assert_called_once_with(mock_websocket)


@pytest.fixture
@pytest.mark.asyncio
@patch("src.api.routers.LlamaCppBackend")
@patch("src.api.routers.StreamingLLMCallbackHandler")
@patch("src.api.routers.manager.connect")
async def test_websocket_endpoint_llamacpp(
    monkeypatch, mock_connect, mock_handler, mock_llamacpp
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key-test")
    monkeypatch.setenv("ENDPOINT_TYPE", "llamacpp")
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.return_value = "Hello, world!"
    await websocket_endpoint(mock_websocket)
    mock_websocket.send_json.assert_called()
    mock_connect.assert_called_once_with(mock_websocket)


@pytest.fixture
@pytest.mark.asyncio
@patch("src.api.routers.manager.connect")
@patch("src.api.routers.manager.disconnect")
async def test_websocket_endpoint_handles_disconnect(
    monkeypatch, mock_disconnect, mock_connect
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key-test")
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()
    await websocket_endpoint(mock_websocket)
    mock_connect.assert_called_once_with(mock_websocket)
    mock_disconnect.assert_called_once_with(mock_websocket)


@pytest.fixture
@pytest.mark.asyncio
@patch("src.api.routers.manager.connect")
async def test_websocket_endpoint_handles_exception(monkeypatch, mock_connect):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key-test")
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = Exception()
    await websocket_endpoint(mock_websocket)
    mock_websocket.send_json.assert_called()
    mock_connect.assert_called_once_with(mock_websocket)
