# vidprom-scorer

Score and rewrite video generation prompts. Built on [VidProM](https://huggingface.co/datasets/Arlene/VidProM) — 100k real text-to-video prompts with metadata.

Multi-output Random Forest on 68 engineered features. Predicts 4 quality dimensions simultaneously.

| Metric | MAE | R² |
|---|---|---|
| specificity | 0.232 | 0.873 |
| clarity | 0.228 | 0.858 |
| visual_richness | 0.351 | 0.697 |
| overall | 0.288 | 0.804 |

## Pipeline

1. [01_labeling.ipynb](notebooks/01_labeling.ipynb) — stratified sampling across 10 clusters, manual quality annotation (250 prompts)
2. [02_features.ipynb](notebooks/02_features.ipynb) — 68 features: lexical, domain (camera/style/lighting), NSFW, sentence embeddings (PCA 50-dim)
3. [03_modeling.ipynb](notebooks/03_modeling.ipynb) — Linear Regression → Random Forest → XGBoost, multi-output RF selected

## API

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

That's all you need to get the scorer running.

### POST /score

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat on the moon"}'
```

```json
{
  "specificity": 2.3,
  "clarity": 4.1,
  "visual_richness": 1.8,
  "overall": 2.7,
  "weak_areas": ["specificity", "visual_richness"]
}
```

### POST /rewrite

Scores the prompt, rewrites weak areas via GPT, re-scores the result.

```bash
curl -X POST http://localhost:8000/rewrite \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A girl entering a glowing portal inside a forest"}'
```

```json
{
  "original_score": 2.27,
  "rewritten": "A young girl with flowing chestnut hair steps toward a swirling iridescent portal...",
  "new_score": 4.08,
  "improvements": ["specificity", "visual_richness"]
}
```

### GET /examples

```bash
curl http://localhost:8000/examples?n=3
```

Returns top best and worst prompts from the labeled dataset with scores.

## Key Insight

`word_count` is the single most important feature at 59.5% importance. Longer, more detailed prompts consistently score higher across all dimensions.
