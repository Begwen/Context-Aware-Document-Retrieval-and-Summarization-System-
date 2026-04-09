import umap
from sklearn.mixture import GaussianMixture

def cluster_embeddings(embeddings, n_components=5):
    reducer = umap.UMAP(n_components=2)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(reduced_embeddings)
    
    return labels, reduced_embeddings
