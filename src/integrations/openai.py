import os
from langchain.chains import ConversationChain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chat_models import ChatOpenAI

from src.utils.logger import get_logger
from src.schemas.api_settings import AbstractAPISettings
from src.utils.callbacks import StreamingLLMCallbackHandler
from src.integrations.llms import BaseLLM


class OpenAISettings(AbstractAPISettings):
    @classmethod
    def from_defaults(cls):
        return OpenAISettings.gpt3_defaults()

    @classmethod
    def gpt3_defaults(cls):
        settings = OpenAISettings()
        settings.type = "openai"
        settings.key = os.environ.get("OPENAI_API_KEY")
        settings.model = "gpt-3.5-turbo"
        settings.max_tokens = 2048
        settings.temperature = 0.3
        return settings


class OpenAIBackend(BaseLLM):
    def __init__(self, api_settings: OpenAISettings = None):
        if api_settings is None:
            self.api_settings = OpenAISettings.from_defaults()
        else:
            self.api_settings = api_settings

    def get_chain(
        self, stream_hanlder: StreamingLLMCallbackHandler
    ) -> ConversationChain:
        """
        Retrieves a ConversationChain object for streaming LLM.
        """
        # Create a stream callback manager
        stream_manager = AsyncCallbackManager([stream_hanlder])

        streaming_llm = ChatOpenAI(
            model_name=self.api_settings.model,
            temperature=self.api_settings.temperature,
            max_tokens=self.api_settings.max_tokens,
            streaming=True,
            callback_manager=stream_manager,
            verbose=True,
        )

        logger = get_logger(__name__)
        logger.info(
            "Created OpenAI Streaming with model name {}, temperature {}, max_tokens {}".format(
                self.api_settings.model,
                self.api_settings.temperature,
                self.api_settings.max_tokens,
            )
        )

        conversation_chain = self.build_conversation_chain(streaming_llm)

        return conversation_chain


class OpenAIFineTunedBackend(OpenAIBackend):
    def __init__(self, model_name: str):
        super().__init__()
        self.api_settings.model = model_name
