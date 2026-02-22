from fastapi import APIRouter

from app.schemas import ScoreResponse, PromptRequest
from app.services.scorer import score_prompt

router = APIRouter()

@router.post("/score", response_model=ScoreResponse)
def score(req: PromptRequest):
    result = score_prompt(req.prompt)
    return ScoreResponse(
        **result["scores"],
        weak_areas=result["weak_areas"]
    )