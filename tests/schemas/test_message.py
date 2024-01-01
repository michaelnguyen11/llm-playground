import pytest
from pydantic import ValidationError
from src.schemas.message import ChatResponse

def test_chat_response_valid():
    # Test with valid data
    response = ChatResponse(sender="bot", message="Hello, world!", type="start")
    assert response.sender == "bot"
    assert response.message == "Hello, world!"
    assert response.type == "start"

def test_chat_response_invalid_sender():
    # Test with invalid sender
    with pytest.raises(ValidationError):
        ChatResponse(sender="invalid", message="Hello, world!", type="start")

def test_chat_response_invalid_type():
    # Test with invalid type
    with pytest.raises(ValidationError):
        ChatResponse(sender="bot", message="Hello, world!", type="invalid")

def test_chat_response_empty_message():
    # Test with empty message
    response = ChatResponse(sender="bot", message="", type="start")
    assert response.sender == "bot"
    assert response.message == ""
    assert response.type == "start"
