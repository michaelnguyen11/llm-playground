from unittest.mock import Mock, patch
from src.integrations.openai import (
    OpenAIBackend,
    OpenAISettings,
)


class TestOpenAIBackend:
    def setup_method(self):
        self.openai_backend = OpenAIBackend()

    def test_init_with_default_settings(self):
        assert isinstance(self.openai_backend.api_settings, OpenAISettings)

    @patch("src.integrations.openai.OpenAISettings.from_defaults")
    def test_init_with_custom_settings(self, mock_settings):
        mock_settings.return_value = "mock_settings"
        openai_backend = OpenAIBackend(api_settings="mock_settings")
        assert openai_backend.api_settings == "mock_settings"

    @patch("src.integrations.openai.AsyncCallbackManager")
    @patch("src.integrations.openai.ChatOpenAI")
    @patch("src.integrations.openai.OpenAIBackend.build_conversation_chain")
    def test_get_chain(
        self,
        mock_build_conversation_chain,
        mock_chat_openai,
        mock_async_callback_manager,
    ):
        mock_stream_handler = Mock()
        mock_async_callback_manager.return_value = "mock_manager"
        mock_chat_openai.return_value = "mock_streaming_llm"
        mock_build_conversation_chain.return_value = "mock_conversation_chain"

        result = self.openai_backend.get_chain(mock_stream_handler)

        assert result == "mock_conversation_chain"
        mock_async_callback_manager.assert_called_once_with([mock_stream_handler])
        mock_chat_openai.assert_called_once_with(
            model_name=self.openai_backend.api_settings.model,
            temperature=self.openai_backend.api_settings.temperature,
            max_tokens=self.openai_backend.api_settings.max_tokens,
            streaming=True,
            callback_manager="mock_manager",
            verbose=True,
        )
        mock_build_conversation_chain.assert_called_once_with("mock_streaming_llm")
