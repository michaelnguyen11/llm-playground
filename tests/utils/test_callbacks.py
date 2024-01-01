# FILEPATH: /Users/hiep/Desktop/Workspace/aitomatic_test/llm-playground/tests/utils/test_callbacks.py

import pytest
from unittest.mock import AsyncMock
from src.utils.callbacks import StreamingLLMCallbackHandler
from src.schemas.message import ChatResponse

@pytest.mark.asyncio
async def test_on_llm_new_token():
    # Mock the websocket
    mock_websocket = AsyncMock()

    # Create an instance of the handler
    handler = StreamingLLMCallbackHandler(mock_websocket)

    # Call the on_llm_new_token method
    await handler.on_llm_new_token('test_token')

    # Create the expected response
    expected_resp = ChatResponse(sender="bot", message='test_token', type="stream")

    # Check that the websocket's send_json method was called with the correct argument
    mock_websocket.send_json.assert_awaited_once_with(expected_resp.dict())
