import sys
from unittest.mock import patch, MagicMock, call
from src.utils.logger import setup_logging, APP_LOGGER_NAME

@patch('logging.getLogger')
@patch('logging.StreamHandler')
def test_setup_logging_default(mock_stream_handler, mock_get_logger):
    # Mock the logger and stream handler
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_handler = MagicMock()
    mock_stream_handler.return_value = mock_handler

    # Call the function
    result = setup_logging()

    # Check that the mocks were called with the correct arguments
    mock_get_logger.assert_called_once_with(APP_LOGGER_NAME)
    mock_stream_handler.assert_called_once_with(sys.stdout)
    mock_logger.addHandler.assert_called_once_with(mock_handler)

    # Check that the result is the mocked logger
    assert result == mock_logger

@patch('logging.getLogger')
@patch('logging.StreamHandler')
@patch('logging.FileHandler')
def test_setup_logging_with_file(mock_file_handler, mock_stream_handler, mock_get_logger):
    # Mock the logger and handlers
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_stream_handler_instance = MagicMock()
    mock_stream_handler.return_value = mock_stream_handler_instance
    mock_file_handler_instance = MagicMock()
    mock_file_handler.return_value = mock_file_handler_instance

    # Call the function
    result = setup_logging(file_name='test.log')

    # Check that the mocks were called with the correct arguments
    mock_get_logger.assert_called_once_with(APP_LOGGER_NAME)
    mock_stream_handler.assert_called_once_with(sys.stdout)
    mock_file_handler.assert_called_once_with('test.log')

    # Check that the logger's addHandler method was called with the correct arguments
    calls = [call(mock_stream_handler_instance), call(mock_file_handler_instance)]
    mock_logger.addHandler.assert_has_calls(calls)

    # Check that the result is the mocked logger
    assert result == mock_logger
