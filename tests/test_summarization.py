from src.summarization import summarize_text

def test_summarization():
    text = """The Eiffel Tower is one of the most famous landmarks in the world. It was constructed in 1889 and is located in Paris, France."""
    summary = summarize_text(text)
    assert isinstance(summary, str) and len(summary) > 10
