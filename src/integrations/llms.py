from abc import abstractmethod, ABC
from langchain.chains import ConversationChain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from src.utils.callbacks import StreamingLLMCallbackHandler


class BaseLLM(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_chain(
        self, stream_hanlder: StreamingLLMCallbackHandler
    ) -> ConversationChain:
        """
        Retrieves a ConversationChain object for streaming LLM.
        """
        pass

    def build_conversation_chain(self, streaming_llm) -> ConversationChain:
        """
        Builds a conversation chain for interacting with a streaming LLM.
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
        # create memory buffer for the conversation
        memory = ConversationBufferMemory(return_messages=True)

        conversation_chain = ConversationChain(
            callback_manager=manager,
            memory=memory,
            llm=streaming_llm,
            verbose=True,
            prompt=prompt,
        )

        return conversation_chain
