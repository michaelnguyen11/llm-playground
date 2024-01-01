import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from src.api.routers import router
from src.utils.logger import setup_logging, get_logger
from src.integrations.openai import settings

logger = setup_logging()

templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
templates = Jinja2Templates(directory=templates_dir)
# app = FastAPI()

# Core Application Instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(router)

@app.get("/")
async def root(requests: Request):
    return templates.TemplateResponse("index.html", {"request": requests})


@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
