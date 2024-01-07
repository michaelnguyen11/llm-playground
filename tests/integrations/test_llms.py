from unittest.mock import Mock, patch
from src.integrations.llms import BaseLLM
from src.integrations.llms import (
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)


class MockBaseLLM(BaseLLM):
    def get_chain(self):
        pass


class TestBaseLLM:
    def setup_method(self):
        self.base_llm = MockBaseLLM()

    @patch("src.integrations.llms.ConversationChain")
    @patch("src.integrations.llms.AsyncCallbackManager")
    @patch("src.integrations.llms.ConversationBufferMemory")
    @patch("src.integrations.llms.ChatPromptTemplate.from_messages")
    def test_build_conversation_chain(
        self,
        mock_chat_prompt_template,
        mock_conversation_buffer_memory,
        mock_async_callback_manager,
        mock_conversation_chain,
    ):
        mock_streaming_llm = Mock()
        mock_chat_prompt_template.return_value = "mock_prompt"
        mock_conversation_buffer_memory.return_value = "mock_memory"
        mock_async_callback_manager.return_value = "mock_manager"
        mock_conversation_chain.return_value = "mock_conversation_chain"

        result = self.base_llm.build_conversation_chain(mock_streaming_llm)

        assert result == "mock_conversation_chain"
        mock_chat_prompt_template.assert_called_once_with(
            [
                SystemMessagePromptTemplate.from_template(
                    "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        mock_conversation_buffer_memory.assert_called_once_with(return_messages=True)
        mock_async_callback_manager.assert_called_once_with([])
        mock_conversation_chain.assert_called_once_with(
            callback_manager="mock_manager",
            memory="mock_memory",
            llm=mock_streaming_llm,
            verbose=True,
            prompt="mock_prompt",
        )
