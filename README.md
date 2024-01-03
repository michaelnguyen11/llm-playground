# llm-playground
The virtual assistant application leverages LLM models, which use FastAPI for the backend and simple Tailwind CSS for the UI.

In the project, I used Langchain build to a conversation chain with memory and OpenAI's GPT-3.5-turbo-1106 model, as well as the fine-tuned GPT-3.5-turbo-1106 model for a specific documents/domain.

## Features
- Dataset generation and fine-tuning OpenAI's GPT for a specific documents/domain in [models](models/README.md)
- Q/A chat with conversation memory using Langchain and OpenAI's GPT models.
- FastAPI websocket endpoint for the backend.
- Simple Tailwind CSS for the UI.
- Dockerized the application using Docker Compose.

## Folder Structure

```bash
llm-playground
    ├───models <- dataset generation and fine-tuning GPT models.
    ├───src <- chat application using FastAPI with a WebSocket endpoint to interact with the GPT models.
    │   ├───api <- define router endpoints.
    │   ├───integrations <- integrate LLM backends like OpenAI or LLaMA.cpp or Intel transformers.
    │   ├───schemas <- define schemas used in the project.
    │   ├───templates <- Tailwind CSS UI.
    │   └───utils <- ultility scripts.
    └───tests <- unittests for the project.
```

## Installation

### Requirements
- Python 3.10
- Docker & Docker compose ([Get Docker](https://docs.docker.com/get-docker/))
- Open AI [API key](https://platform.openai.com/account/api-keys)

### Setup

```bash
git clone https://github.com/michaelnguyen11/llm-playground.git
cd llm-playground
cp .env.template .env

# Edit your .env file
```

### Docker-compose
The easiest way to get started is using the docker-compose, it will Dockerized the application.
```
docker-compose build
docker-compose up -d
```
Then, navigate to `http://0.0.0.0:8080` to chat with the Q/A virtual assistant

### Local Python Environment
Encourage to run in the virtual environments like `venv` or `conda`. To install dependency packages, use the command :
```
pip install -r requirements.txt
```

To launch the server, use the command:
```
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```
Then, navigate to `http://0.0.0.0:8080` to chat with the Q/A virtual assistant


## Potential areas for improvement.
1. Improve the data generation pipeline
2. Explore more methods to evaluate fine-tuned LLM models.
3. Implement RAG with Langchain for augmenting LLM knowledge with additional data
4. Multiple LLM backends for LocalLLM, optimized for specific Hardware
- [ ] [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) : MacOS Platforms
- [ ] [intel-extension-for-transformers](https://github.com/intel/intel-extension-for-transformers) : Intel Platforms
- [ ] [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) : Nvidia Platforms
