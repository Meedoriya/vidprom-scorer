import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "score_model.pkl"

STYLE_KEYWORDS = ['cinematic', 'anime', 'realistic', 'photorealistic',
                  '8k', '4k', 'hd', 'uhd', 'ultra', 'hyperrealistic']
CAMERA_KEYWORDS = ['close-up', 'closeup', 'aerial', 'slow motion',
                   'slowmo', 'pan', 'zoom', 'tracking shot', 'wide shot',
                   'bird', 'eye level', 'handheld']
LIGHTING_KEYWORDS = ['golden hour', 'dramatic lighting', 'neon', 'sunlight',
                     'moonlight', 'shadow', 'backlit', 'rim light', 'fog', 'haze']
COLOR_KEYWORDS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange',
                  'black', 'white', 'dark', 'bright', 'vibrant', 'colorful',
                  'monochrome', 'pastel']


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["features"], bundle["targets"]

model, scaler, feature_names, target_names = load_model()


def extract_features(prompt: str) -> pd.DataFrame:
    text = str(prompt)
    t = text.lower()
    words = t.split()

    basic = {
        'word_count': len(words),
        'char_count': len(text),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'comma_count': text.count(','),
        'has_numbers': int(bool(re.search(r'\d', text))),
    }

    domain = {
        'has_style': int(any(k in t for k in STYLE_KEYWORDS)),
        'has_camera': int(any(k in t for k in CAMERA_KEYWORDS)),
        'has_lighting': int(any(k in t for k in LIGHTING_KEYWORDS)),
        'has_color': int(any(k in t for k in COLOR_KEYWORDS)),
        'domain_score': sum([
            any(k in t for k in STYLE_KEYWORDS),
            any(k in t for k in CAMERA_KEYWORDS),
            any(k in t for k in LIGHTING_KEYWORDS),
            any(k in t for k in COLOR_KEYWORDS)
        ]),
    }

    nsfw = {col: 0.0 for col in [
        'toxicity', 'obscene', 'identity_attack',
        'insult', 'threat', 'sexual_explicit'
    ]}
    nsfw['nsfw_max'] = 0.0
    nsfw['nsfw_sum'] = 0.0

    emb = {f'emb_{i}': 0.0 for i in range(50)}

    row = {**basic, **domain, **nsfw, **emb}
    df = pd.DataFrame([row])[feature_names]
    return df


def score_prompt(prompt: str) -> dict:
    X = extract_features(prompt)
    pred = model.predict(X)[0]

    scores = {name: round(float(pred[i]), 2) for i, name in enumerate(target_names)}

    threshold = 3.0
    weak = [name for name, val in scores.items()
            if val < threshold and name != "overall"]
    return {"scores": scores, "weak_areas": weak}