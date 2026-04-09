from transformers import pipeline

def summarize_text(text, model_name='facebook/bart-large-cnn'):
    summarizer = pipeline("summarization", model=model_name)
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
