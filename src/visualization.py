import matplotlib.pyplot as plt

def plot_clusters(reduced_embeddings, labels):
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()
