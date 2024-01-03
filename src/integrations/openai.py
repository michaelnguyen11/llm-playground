from langchain.chains import ConversationChain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseSettings
from src.utils.logger import get_logger

class Settings(BaseSettings):
    PROJECT_NAME: str = "llm-playground"
    API_VERSION: str = "v1"
    API_V1_STR: str = f"/api/{API_VERSION}"
    OPENAI_API_KEY: str = "sk-..."
    MODEL_NAME: str = "gpt-3.5-turbo"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000


settings = Settings()

def get_openai_chain(stream_hanlder, tracing: bool = False) -> ConversationChain:
    """
    Create a ConversationChain for question/answering
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # Create a callback manager
    manager = AsyncCallbackManager([])
    # Create a stream callback manager
    stream_manager = AsyncCallbackManager([stream_hanlder])

    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    streaming_llm = ChatOpenAI(
        model_name=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
    )
    logger = get_logger(__name__)
    logger.info("Created OpenAI Streaming with model_name {}, temperature {}, max_tokens {}".format(settings.MODEL_NAME, settings.TEMPERATURE, settings.MAX_TOKENS))

    memory = ConversationBufferMemory(return_messages=True)

    conversation_chain = ConversationChain(
        callback_manager=manager, memory=memory, llm=streaming_llm, verbose=True, prompt=prompt
    )

    return conversation_chain
