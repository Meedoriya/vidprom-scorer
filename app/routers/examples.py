from pathlib import Path

import pandas as pd

from fastapi import APIRouter

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "labeled_prompts.parquet"
df = pd.read_parquet(DATA_PATH)

router = APIRouter()

@router.get("/examples")
def examples(n: int = 5):
    best = df.nlargest(n, "overall")[["prompt", "specificity", "clarity", "visual_richness", "overall"]]
    worst = df.nsmallest(n, "overall")[["prompt", "specificity", "clarity", "visual_richness", "overall"]]

    return {
        "best": best.to_dict(orient="records"),
        "worst": worst.to_dict(orient="records")
    }