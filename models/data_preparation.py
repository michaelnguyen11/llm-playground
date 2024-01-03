import os
from typing import List
import openai
from llama_index import SimpleDirectoryReader, ServiceContext, Document, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator
from llama_index.callbacks import CallbackManager, OpenAIFineTuningHandler

from utils import load_documents

openai.api_key = os.environ["OPENAI_API_KEY"]

def question_generator(
        service_context: ServiceContext,
        documents: List[Document],
        question_gen_query: tuple,
        number_generated_questions: int,
        output_file: str):
    """
    Generate questions from documents using OpenAI GPT models
    """

    dataset_generator = DatasetGenerator.from_documents(
        documents=documents,
        question_gen_query=question_gen_query,
        service_context=service_context,
        num_questions_per_chunk=25,
    )

    questions = dataset_generator.generate_questions_from_nodes(num=number_generated_questions)
    print("Generated {} questions".format(len(questions)))

    with open(output_file, "a+") as f:
        for question in questions:
            f.write(question + "\n")

def questions_generator(
        service_context: ServiceContext,
        documents: SimpleDirectoryReader,
        question_gen_query: tuple,
        train_questions_file: str='datasets/train_questions.txt',
        eval_questions_file:str ='datasets/eval_questions.txt'):

    """
    Generate questions from documents using OpenAI GPT models
    """
    # question_generator(service_context, documents[:15], question_gen_query, 200, train_questions_file)
    # question_generator(service_context, documents[15:], question_gen_query, 150, eval_questions_file)
    question_generator(service_context, documents, question_gen_query, 200, train_questions_file)

def dataset_generator(
        documents: SimpleDirectoryReader, 
        train_questions_file='datasets/train_questions.txt', 
        output_finetuning_file='datasets/finetuning_events.jsonl'):
    """
    Generate dataset for fine-tuning from generated questions using GPT-4
    """

    # Generate answers from train_questions.txt using GPT-4 
    finetuning_handler = OpenAIFineTuningHandler()
    callback_manager = CallbackManager([finetuning_handler])

    gpt_4_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-4-1106-preview", temperature=0.3),
        context_window=2048,  # limit the context window artifically to test refine process
        callback_manager=callback_manager,
    )

    questions = []
    with open(train_questions_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=gpt_4_context
    )

    query_engine = index.as_query_engine(similarity_top_k=2)
    for question in questions:
        response = query_engine.query(question)
        print(response)

    finetuning_handler.save_finetuning_events(output_finetuning_file)

if __name__ == '__main__':
    documents = load_documents(["docs/Generative_Agents_Interactive_Simulacra_of_Human_Behavior.pdf"])
    question_gen_query = (
        "You are a Teacher/ Professor. Your task is to setup "
        "a examination. Using the provided context, formulate "
        "a single question that captures an important fact from the "
        "context. Restrict the question to the context information provided."
    )

    # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.3))
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-1106-preview", temperature=0.3))
    
    questions_generator(service_context,
                        documents,
                        question_gen_query,
                        train_questions_file='datasets/train_questions_gpt4_generate.txt',
                        eval_questions_file='datasets/eval_questions_gpt4_generate.txt')

    dataset_generator(documents,
                      train_questions_file='datasets/train_questions_gpt4_generate.txt',
                      output_finetuning_file='datasets/finetuning_events_gpt4_100_questions.jsonl')
