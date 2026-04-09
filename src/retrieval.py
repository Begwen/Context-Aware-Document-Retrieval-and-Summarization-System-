import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_similar(query_embedding, doc_embeddings, top_n=3):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return top_indices
