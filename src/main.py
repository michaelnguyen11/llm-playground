import os
import dotenv
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from src.api.routers import router
from src.utils.logger import setup_logging, get_logger

dotenv.load_dotenv(override=True)

logger = setup_logging()

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

app = FastAPI(title="llm-playground")

app.include_router(router)


@app.get("/")
async def root(requests: Request):
    return templates.TemplateResponse("index.html", {"request": requests})


@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
