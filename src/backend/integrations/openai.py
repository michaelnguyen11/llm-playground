from langchain.chains import ConversationChain
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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

    streaming_llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2000,
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
    )

    memory = ConversationBufferMemory(return_messages=True)

    conversation_chain = ConversationChain(
        callback_manager=manager, memory=memory, llm=streaming_llm, verbose=True, prompt=prompt
    )

    return conversation_chain
