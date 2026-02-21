import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)

df = pd.read_parquet("data/labeled_prompts.parquet")

print("Computing embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
raw_embeddings = model.encode(df['prompt'].tolist(), show_progress_bar=True)
print(f"Raw shape: {raw_embeddings.shape}")

pca = PCA(n_components=50, random_state=SEED)
embeddings = pca.fit_transform(raw_embeddings)
print(f"PCA shape: {embeddings.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

np.save("data/embeddings_250.npy", embeddings)
print("Saved: data/embeddings_250.npy")