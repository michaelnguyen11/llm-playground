# Fine-Tuning OpenAI's GPT-3.5 Turbo from documents
In this repository, I want to extract knowledge from the documents to create a custom dataset using GPT-4 for fine-tuning the GPT-3.5.
The aim is to distill knowledge from GPT-4 to GPT-3.5 so that a smaller GPT-3.5 becomes closer to GPT-4 performance in specific subject.

## Fine-tuning results
With the dataset in `datasets` folder, the `answer_relevancy` score and `faithfulness` score of `ragas` evaluation framework increased dramatically in the fine-tuned GPT-3.5-turbo-1106 model, compare to the baseline GPT-3.5-turbo-1106 model, nearly reach to GPT-4-1106-preview performance.

Model Name | Answer Relevancy Score| Faithfulness Score
--- | --- | ---
GPT-3.5-turbo-1106 | 0.8806072429800716 | 0.8691269841269841
Fine-tuned GPT-3.5-turbo-1106 | 0.9030882416787889 | 0.9191919191919191
gpt-4-1106-preview | 0.9067527587830269 | 0.9327409627409627


### Setup environment
The neccessary packages for this module is define at `requirements.txt` file.
```
pip install requirements.txt
```

### Generate the dataset
Ensure that you exported the `OPENAI_API_KEY` as an environment variable.

To generate the dataset, it includes 2 steps : 
- Step 1 : generate train/eval questions using GPT-4. Then do the feature engineering to get filter out train and evaluate questions.
- Step 2 : from the generated train questions, we use GPT-4 to generate answers from these questions.
Then do the feature engineering to get filter out GPT-4's answers these questions, to get the training dataset.

- To generate train/eval questions using GPT-4
```
python3 data_preparation.py --question --train_path /path/to/train_questions.txt --val_path /path/to/eval_questions.txt
```

- To generate training answers using GPT-4 from above questions
```
python3 data_preparation.py --dataset --train_path /path/to/train_questions.txt --finetune_path /path/to/finetune_dataset.jsonl
```

Please check the `python3 data_preparation.py --help` for the `data_preparation.py` usage.

### Fine-tuning
After creating the fine-tuning dataset, launch a fine-tuning job to fine-tune the GPT-3.5-turbo-1106 model:
```
python3 launch_training.py /path/to/finetune_dataset.jsonl
```
It will create a OpenAI's fine-tuning job and output a fine-tune job id. To get the information about a fine-tuneing job, use command :
```
curl https://api.openai.com/v1/fine_tuning/jobs/your_ftjob_id \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Evaluation
For the evaluation, I use the [ragas evaluation framework](https://github.com/explodinggradients/ragas) to evaluate the baseline for GPT-3.5-turbo-1106 model, as well as the fine-tuned GPT-3.5-turbo-1106 model with the generated evaluation questions in previous section.

I will use 2 metrics from `ragas` evaluation framework to evaluate the models:
- `answer_relevancy` - This measures how relevant is the generated answer to the prompt. If the generated answer is incomplete or contains redundant information the score will be low. This is quantified by working out the chance of an LLM generating the given question using the generated answer. Values range (0,1), higher the better.
- `faithfulness` - This measures the factual consistency of the generated answer against the given context. This is done using a multi step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to (0,1) range. Higher the better.


- To evaluate the baseline for GPT-3.5-turbo-1106 model with generated eval questions:
```
python3 launch_evaluation.py --eval_baseline --val_path /path/to/eval_questions.txt
```

- To evaluate fine-tuned GPT-3.5-turbo-1106 model with generated eval questions:
```
python3 launch_evaluation.py --eval_finetuned --val_path /path/to/eval_questions.txt
```

- To evaluate gpt-4-1106-preview model with generated eval questions:
```
python3 launch_evaluation.py --eval_gpt4 --val_path /path/to/eval_questions.txt
```

- To compare the differences in responses between GPT-3.5-turbo-1106, fine-tuned GPT-3.5-turbo-1106 and gpt-4-1106-preview models
```
python3 launch_evaluation.py --compare_response --val_path /path/to/eval_questions.txt --response_file /path/to/compare_responses.csv
```
