from src.embedding import get_embedding

def test_embedding():
    text = "Hello world"
    embedding = get_embedding(text)
    assert embedding is not None
