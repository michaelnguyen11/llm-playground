import os
import openai
import pandas as pd
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from utils import load_documents

openai.api_key = os.environ["OPENAI_API_KEY"]

def evaluation_with_ragas(service_context, documents, eval_questions_file):
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

def evaluate_gpt_model(documents, model_name="gpt-3.5-turbo-1106", eval_questions_file="datasets/eval_questions.txt"):
    # limit the context window to 2048 tokens so that refine is used
    gpt_35_context = ServiceContext.from_defaults(
        llm=OpenAI(model=model_name, temperature=0.3),
        context_window=2048,
    )

    evaluation_result = evaluation_with_ragas(gpt_35_context, documents, eval_questions_file)

    return evaluation_result

def explore_responses(documents, model_name, questions_file, question_number=11):
    index = VectorStoreIndex.from_documents(documents)
    questions = []
    with open(questions_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    print(questions[question_number])

    gpt_35_context = ServiceContext.from_defaults(
        llm=OpenAI(model=model_name, temperature=0.3),
        context_window=2048,  # limit the context window artifically to test refine process
    )
    query_engine = index.as_query_engine(service_context=gpt_35_context)
    response = query_engine.query(questions[question_number])

    return response


if __name__ == '__main__':
    documents = load_documents(["docs/generative_agent.pdf"])

    gpt_35_baseline = 'gpt-3.5-turbo-1106'
    gpt_35_tuned = 'ft:gpt-3.5-turbo-1106:aitomatic-inc:hiep:8clVbyfK'
    gpt_4_baseline = 'gpt-4'

    evaluation_result_baseline = evaluate_gpt_model(documents, gpt_35_baseline)
    evaluation_result_tuned = evaluate_gpt_model(documents, gpt_35_tuned)
    gpt4_result_baseline = evaluate_gpt_model(documents, gpt_4_baseline)

    print('Model {} evaluation result: {}'.format(gpt_35_baseline, evaluation_result_baseline))
    print('Model {} evaluation result: {}'.format(gpt_35_tuned, evaluation_result_tuned))
    print('Model {} evaluation result: {}'.format(gpt_4_baseline, gpt4_result_baseline))

    # Let's quickly compare the differences in responses,
    # to demonstrate that fine tuning did indeed change something.
    gpt_35_baseline_response = explore_responses(documents, gpt_35_baseline, "eval_questions.txt", 12)
    print("Model {} - Final Response: {}".format(gpt_35_baseline, gpt_35_baseline_response.response.strip()))
    gpt_35_tuned_response = explore_responses(documents, gpt_35_tuned, "eval_questions.txt", 12)
    print('Model {} - Final Response: {}'.format(gpt_35_tuned, gpt_35_tuned_response.response.strip()))
    gpt_4_baseline_response = explore_responses(documents, gpt_4_baseline, "eval_questions.txt", 12)
    print("Model {} - Final Response: {}".format(gpt_4_baseline, gpt_4_baseline_response.response.strip()))
