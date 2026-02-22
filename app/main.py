from fastapi import FastAPI

from app.routers import score, rewrite, examples

app = FastAPI(
    title = "VidProm Scorer",
    description="Score and rewrite video generation prompts",
    version="0.1.0"
)

app.include_router(score.router)
app.include_router(rewrite.router)
app.include_router(examples.router)