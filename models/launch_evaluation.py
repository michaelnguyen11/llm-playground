import os
import openai
import pandas as pd
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from utils import load_documents

openai.api_key = os.environ["OPENAI_API_KEY"]

def evaluation_with_ragas(
        service_context: ServiceContext,
        documents: SimpleDirectoryReader,
        eval_questions_file: str):
    """
    Evaluate the model with Ragas
    """
    questions = []
    with open(eval_questions_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context
    )

    query_engine = index.as_query_engine(similarity_top_k=2)

    contexts = []
    answers = []

    for question in questions:
        response = query_engine.query(question)
        contexts.append([x.node.get_content() for x in response.source_nodes])
        answers.append(str(response))

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    result = evaluate(ds, [answer_relevancy, faithfulness])

    return result


def evaluate_gpt_model(
        documents: SimpleDirectoryReader,
        model_name: str,
        eval_questions_file: str):
    """
    Evaluate the GPT models with Ragas
    """
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=model_name, temperature=0.3),
        context_window=2048, # limit the context window to 2048 tokens so that refine is used
    )

    result = evaluation_with_ragas(service_context, documents, eval_questions_file)

    return result


def get_question(questions_file: str, question_number: str):
    """
    Get question from questions file
    """
    questions = []
    with open(questions_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    question = questions[question_number]

    return question


def get_response(
        documents: SimpleDirectoryReader,
        model_name: str,
        question: int):
    """
    Get response from model from a specific question
    """
    index = VectorStoreIndex.from_documents(documents)
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=model_name, temperature=0.3),
        context_window=2048,  # limit the context window artifically to test refine process
    )
    query_engine = index.as_query_engine(service_context=service_context)
    answer = query_engine.query(question)

    return answer


if __name__ == '__main__':
    documents = load_documents(["docs/Generative_Agents_Interactive_Simulacra_of_Human_Behavior.pdf"])

    gpt_35_baseline = 'gpt-3.5-turbo-1106'
    # The fine-tuned model trained with train questions generated from GPT-3.5-turbo-1106
    # gpt_35_tuned = 'ft:gpt-3.5-turbo-1106:aitomatic-inc:hiep:8clVbyfK'
    # The fine-tuned model trained with train questions generated from GPT-4
    gpt_35_tuned = 'ft:gpt-3.5-turbo-1106:aitomatic-inc:hiep:8cuu5f77'
    gpt_4_baseline = 'gpt-4-1106-preview'

    gpt_35_baseline_result = evaluate_gpt_model(documents=documents,
                                                model_name=gpt_35_baseline,
                                                eval_questions_file="datasets/eval_questions_gpt4_generate.txt")
    gpt_35_tuned_result = evaluate_gpt_model(documents=documents,
                                             model_name=gpt_35_tuned,
                                             eval_questions_file="datasets/eval_questions_gpt4_generate.txt")
    gpt_4_baseline_result = evaluate_gpt_model(documents=documents,
                                               model_name=gpt_4_baseline,
                                               eval_questions_file="datasets/eval_questions_gpt4_generate.txt")

    print('Evaluation model {} with Ragas : {}'.format(gpt_35_baseline, gpt_35_baseline_result))
    print('Evaluation model {} with Ragas : {}'.format(gpt_35_tuned, gpt_35_tuned_result))
    print('Evaluation model {} with Ragas : {}'.format(gpt_4_baseline, gpt_4_baseline_result))

    question = get_question(questions_file="datasets/eval_questions_gpt4_generate.txt", question_number=12)

    gpt_35_answer = get_response(documents=documents,
                                 model_name=gpt_35_baseline,
                                 question=question)
    gpt_35_tuned_answer = get_response(documents=documents,
                                       model_name=gpt_35_tuned,
                                       question=question)
    gpt_4_baseline_answer = get_response(documents=documents,
                                         model_name=gpt_4_baseline,
                                         question=question)

    # Let's quickly compare the differences in responses,
    # to demonstrate that fine tuning did indeed change something.
    eval_df = pd.DataFrame(
        {
            "Question": question,
            "Model Name": [gpt_35_baseline, gpt_35_tuned, gpt_4_baseline],
            "Answer": [gpt_35_answer, gpt_35_tuned_answer, gpt_4_baseline_answer],
            "Ragas Answer Selevancy Score": [gpt_35_baseline_result['answer_relevancy'],
                                             gpt_35_tuned_result['answer_relevancy'],
                                             gpt_4_baseline_result['answer_relevancy']],
            "Ragas Faithfulness Score": [gpt_35_baseline_result['faithfulness'],
                                         gpt_35_tuned_result['faithfulness'],
                                         gpt_4_baseline_result['faithfulness']],
        },
    )

    eval_df.to_csv('compare_responses.csv', index=False)
    print(eval_df)
