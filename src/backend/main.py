from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from api.routers import router

templates = Jinja2Templates(directory="templates")
app = FastAPI()
app.include_router(router)

@app.get("/")
async def root(requests: Request):
    return templates.TemplateResponse("index.html", {"request": requests})


@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
