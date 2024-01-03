import pytest
import os
from fastapi import WebSocketDisconnect
from src.api.routers import websocket_endpoint, manager
from unittest.mock import AsyncMock

from unittest.mock import patch

### Remind to read again Copilot chat for the test_routers.py file

os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
@pytest.mark.asyncio
@patch('src.api.routers.get_openai_chain')
@patch('src.api.routers.StreamingLLMCallbackHandler')
@patch('src.api.routers.manager.connect')
async def test_websocket_endpoint_receives_and_sends_text(mock_connect, mock_handler, mock_chain):
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.return_value = "Hello, world!"
    await websocket_endpoint(mock_websocket)
    mock_websocket.send_json.assert_called()
    mock_connect.assert_called_once_with(mock_websocket)

@pytest.mark.asyncio
@patch('src.api.routers.get_openai_chain')
@patch('src.api.routers.StreamingLLMCallbackHandler')
@patch('src.api.routers.manager.connect')
@patch('src.api.routers.manager.disconnect')
async def test_websocket_endpoint_handles_disconnect(mock_disconnect, mock_connect, mock_handler, mock_chain):
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = WebSocketDisconnect()
    await websocket_endpoint(mock_websocket)
    mock_connect.assert_called_once_with(mock_websocket)
    mock_disconnect.assert_called_once_with(mock_websocket)

@pytest.mark.asyncio
@patch('src.api.routers.get_openai_chain')
@patch('src.api.routers.StreamingLLMCallbackHandler')
@patch('src.api.routers.manager.connect')
async def test_websocket_endpoint_handles_exception(mock_connect, mock_handler, mock_chain):
    mock_websocket = AsyncMock()
    mock_websocket.receive_text.side_effect = Exception()
    await websocket_endpoint(mock_websocket)
    mock_websocket.send_json.assert_called()
    mock_connect.assert_called_once_with(mock_websocket)