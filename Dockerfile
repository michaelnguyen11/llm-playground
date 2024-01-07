FROM python:3.10-slim

WORKDIR /app

COPY . /app
RUN pip install $(grep -ivE "llama-cpp-python|huggingface-cli" requirements.txt)

EXPOSE 8080:8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
