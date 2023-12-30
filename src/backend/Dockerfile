FROM python:3.10-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080:8080

# loads server.py module and the FastAPI `app` object
CMD uvicorn server:app --host 0.0.0.0 --port 8080
