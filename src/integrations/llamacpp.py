import platform
from langchain.chains import ConversationChain
from langchain.callbacks.manager import AsyncCallbackManager
# from langchain_community.llms import LlamaCpp
from src.integrations.fix_llamacpp_async import LlamaCpp

from src.utils.logger import get_logger
from src.schemas.api_settings import AbstractAPISettings
from src.utils.callbacks import StreamingLLMCallbackHandler
from src.integrations.llms import BaseLLM


class LlamaCppSettings(AbstractAPISettings):
    @classmethod
    def from_defaults(cls):
        return LlamaCppSettings.llamacpp_defaults()

    @classmethod
    def llamacpp_defaults(cls):
        settings = LlamaCppSettings()
        settings.type = "llamacpp"
        settings.model = "data/llama2-chat-7b/llama-2-7b-chat.Q3_K_S.gguf"
        settings.engine = "cpu"
        settings.max_tokens = 2048
        settings.temperature = 0.3

        if platform.processor() == "arm":
            settings.engine = "metal"
            settings.n_gpu_layers = 1
            settings.n_batch = 8
            settings.f16_kv = True

        # TODO: check if GPU is available, using python standard library only

        return settings


class LlamaCppBackend(BaseLLM):
    def __init__(self, api_settings: LlamaCppSettings = None):
        if api_settings is None:
            self.api_settings = LlamaCppSettings.from_defaults()
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

        streaming_llm = LlamaCpp(
            model_path=self.api_settings.model,
            temperature=self.api_settings.temperature,
            max_tokens=self.api_settings.max_tokens,
            streaming=True,
            callback_manager=stream_manager,
            verbose=True,
        )

        logger = get_logger(__name__)
        logger.info(
            "Created LlamaCpp Streaming with model name {}, temperature {}, max_tokens {}".format(
                self.api_settings.model,
                self.api_settings.temperature,
                self.api_settings.max_tokens,
            )
        )

        conversation_chain = self.build_conversation_chain(streaming_llm)

        return conversation_chain
