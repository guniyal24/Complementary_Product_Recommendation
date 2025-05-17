import numpy as np
from sentence_transformers import SentenceTransformer
import torch

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384

try:
    embedding_model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    embedding_model = None

def get_embedding(text_to_embed):
    if not text_to_embed:
        return None

    if embedding_model is None:
        return None

    try:
        embeddings_np = embedding_model.encode(text_to_embed, convert_to_numpy=True)

        if embeddings_np.ndim == 2 and embeddings_np.shape[0] == 1:
             embedding_list = embeddings_np[0].tolist()
        elif embeddings_np.ndim == 1:
             embedding_list = embeddings_np.tolist()
        else:
             return None

        if len(embedding_list) != VECTOR_DIMENSION:
            return None

        return embedding_list

    except Exception as e:
        return None