# vidprom-scorer

Score and rewrite video generation prompts. Built on [VidProM](https://huggingface.co/datasets/Arlene/VidProM) — 100k real text-to-video prompts with metadata.

Random Forest on 68 engineered features. MAE 0.299, R² 0.807 on held-out test set.

## Pipeline

1. [01_labeling.ipynb](notebooks/01_labeling.ipynb) — stratified sampling across 10 clusters, manual quality annotation (250 prompts: specificity, clarity, visual richness, overall)
2. [02_features.ipynb](notebooks/02_features.ipynb) — feature engineering: lexical, semantic (sentence-transformers + PCA), toxicity, domain signals
3. [03_modeling.ipynb](notebooks/03_modeling.ipynb) — Linear Regression → Random Forest → XGBoost comparison; RF selected as best

## Install

```bash
pip install -r requirements.txt
```