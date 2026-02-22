from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str


class ScoreResponse(BaseModel):
    specificity: float
    clarity: float
    visual_richness: float
    overall: float
    weak_areas: list[str]


class RewriteResponse(BaseModel):
    original_score: float
    rewritten: str
    new_score: float
    improvements: list[str]
