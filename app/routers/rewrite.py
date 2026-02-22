from fastapi import APIRouter

from app.schemas import RewriteResponse, PromptRequest
from app.services.rewriter import rewrite_prompt
from app.services.scorer import score_prompt

router = APIRouter()


@router.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: PromptRequest):
    original = score_prompt(req.prompt)
    rewritten_text = rewrite_prompt(req.prompt, original["weak_areas"])
    new = score_prompt(rewritten_text)

    return RewriteResponse(
        original_score=original["scores"]["overall"],
        rewritten=rewritten_text,
        new_score=new["scores"]["overall"],
        improvements=original["weak_areas"]
    )
