from unittest.mock import Mock, patch
from src.integrations.llamacpp import (
    LlamaCppBackend,
    LlamaCppSettings,
)


class TestLlamaCppBackend:
    def setup_method(self):
        self.llamacpp_backend = LlamaCppBackend()

    def test_init_with_default_settings(self):
        assert isinstance(self.llamacpp_backend.api_settings, LlamaCppSettings)

    @patch("src.integrations.llamacpp.LlamaCppSettings.from_defaults")
    def test_init_with_custom_settings(self, mock_settings):
        mock_settings.return_value = "mock_settings"
        llamacpp_backend = LlamaCppBackend(api_settings="mock_settings")
        assert llamacpp_backend.api_settings == "mock_settings"

    @patch("src.integrations.llamacpp.AsyncCallbackManager")
    @patch("src.integrations.llamacpp.LlamaCpp")
    @patch("src.integrations.llamacpp.LlamaCppBackend.build_conversation_chain")
    def test_get_chain(
        self, mock_build_conversation_chain, mock_llamacpp, mock_async_callback_manager
    ):
        mock_stream_handler = Mock()
        mock_async_callback_manager.return_value = "mock_manager"
        mock_llamacpp.return_value = "mock_streaming_llm"
        mock_build_conversation_chain.return_value = "mock_conversation_chain"

        result = self.llamacpp_backend.get_chain(mock_stream_handler)

        assert result == "mock_conversation_chain"
        mock_async_callback_manager.assert_called_once_with([mock_stream_handler])
        mock_llamacpp.assert_called_once_with(
            model_path=self.llamacpp_backend.api_settings.model,
            temperature=self.llamacpp_backend.api_settings.temperature,
            max_tokens=self.llamacpp_backend.api_settings.max_tokens,
            streaming=True,
            callback_manager="mock_manager",
            verbose=True,
        )
        mock_build_conversation_chain.assert_called_once_with("mock_streaming_llm")
