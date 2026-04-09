from src.clustering import cluster_embeddings
import numpy as np

def test_clustering():
    fake_embeddings = np.random.rand(10, 384)
    labels, _ = cluster_embeddings(fake_embeddings)
    assert len(labels) == 10
