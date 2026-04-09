from src.retrieval import retrieve_similar
import numpy as np

def test_retrieval():
    fake_embeddings = np.random.rand(10, 384)
    query_embedding = fake_embeddings[0]
    top_docs = retrieve_similar(query_embedding, fake_embeddings)
    assert len(top_docs) == 3
