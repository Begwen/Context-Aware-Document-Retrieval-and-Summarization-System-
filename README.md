# SmartSummarizerRAG

### A Hierarchical Retrieval-Augmented Generation System with Recursive Summarization

> **TL;DR** — Feed it any corpus of documents. It clusters them semantically, builds a multi-level summary hierarchy, stores everything in a vector database, and answers natural language questions with precise, context-grounded responses — all running locally with quantized LLMs. No API keys. No cloud dependency. Full privacy.

---

## Why This Project Exists

Traditional RAG systems retrieve raw chunks and hope the LLM can make sense of them. This doesn't scale — when documents number in the hundreds, you lose the forest for the trees.

**SmartSummarizerRAG** solves this with a **recursive embed-cluster-summarize** architecture:

1. Embed all document chunks into a high-dimensional vector space
2. Cluster them using UMAP + Gaussian Mixture Models (unsupervised, auto-tuned)
3. Summarize each cluster using a local LLM
4. Repeat: treat summaries as new documents, embed-cluster-summarize again
5. Store the entire hierarchy (raw chunks + all summary levels) in ChromaDB
6. At query time, retrieve from *any abstraction level* — detail or overview

The result is a system that can answer both "What is LangChain Expression Language?" and "Give me a high-level summary of all documents" from the same vector store.

---

## Architecture

```
                         +-----------------------+
                         |   Document Sources    |
                         | (URLs, PDFs, Text)    |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------+-----------+
                         |  Recursive URL Loader |
                         |  + Token-Aware Split  |
                         |  (tiktoken, 1000 tok) |
                         +-----------+-----------+
                                     |
                          +----------+----------+
                          |                     |
                    Level 1               Level 1
                   Embedding             Embedding
                  (BAAI/bge-large)     (BAAI/bge-large)
                          |                     |
                    +-----v-----+         +-----v-----+
                    |   UMAP    |         |   UMAP    |
                    | + GMM     |         | + GMM     |
                    | (Global)  |         | (Local)   |
                    +-----+-----+         +-----+-----+
                          |                     |
                    +-----v-----+         +-----v-----+
                    | Cluster   |         | Cluster   |
                    | Summaries |         | Summaries |
                    +-----+-----+         +-----+-----+
                          |                     |
                          +-----> Level 2 <-----+
                                Embed + Cluster
                                + Summarize
                                     |
                                     v
                               Level 3 (Meta)
                                     |
              +----------------------+----------------------+
              |                      |                      |
        Raw Chunks            L1 Summaries           L2-L3 Summaries
              |                      |                      |
              +----------+-----------+----------------------+
                         |
                   +-----v-----+
                   | ChromaDB  |
                   | Vector    |
                   | Store     |
                   +-----+-----+
                         |
              +----------+----------+
              |                     |
        +-----v-----+        +-----v-----+
        | Retriever  |        |  Local    |
        | (Top-K     |        |  LLM      |
        |  Cosine)   +------->+ (Zephyr/  |
        +------------+        |  Llama)   |
                              +-----+-----+
                                    |
                                    v
                            +-------+-------+
                            |   Answer      |
                            | (Grounded in  |
                            |  context)     |
                            +---------------+
```

---

## Key Technical Decisions

| Decision | Why It Matters |
|---|---|
| **Recursive Embed-Cluster-Summarize** | Creates a semantic pyramid — retrieval works at any abstraction level, not just raw chunks |
| **Two-Tier Clustering (Global + Local)** | First pass finds macro-themes via UMAP+GMM; second pass finds sub-topics within each. Captures natural document hierarchies |
| **BIC-Based Automatic Cluster Count** | No manual `n_clusters` tuning. GMM's Bayesian Information Criterion finds the optimal number mathematically |
| **Soft Clustering with Probability Threshold** | GMM outputs probabilities — documents can belong to multiple clusters (threshold: 0.1). More realistic than hard k-means |
| **Token-Aware Splitting (tiktoken)** | Chunks respect actual model token limits, not naive character counts |
| **Local Quantized LLMs (GGUF Q4_K_M)** | 4-bit quantization runs 7-8B param models on consumer hardware. Zero API cost, full data privacy |
| **Normalized Embeddings** | Cosine similarity becomes a simple dot product — faster retrieval, consistent distance metric across UMAP and ChromaDB |
| **Hierarchical Vector Store** | Stores raw docs AND summaries. Detail queries hit leaf nodes; overview queries hit summary nodes |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangChain, LangChain Expression Language (LCEL) |
| **Embeddings** | `BAAI/bge-large-en-v1.5` (1024-dim) via Sentence Transformers |
| **Vector Store** | ChromaDB (in-memory / persistent) |
| **LLM Inference** | `llama-cpp-python` with GGUF quantized models |
| **Models** | Zephyr-7B-beta / Meta-Llama-3.1-8B-Instruct (Q4_K_M) |
| **Summarization** | `facebook/bart-large-cnn` (seq2seq) |
| **Clustering** | UMAP (dimensionality reduction) + Gaussian Mixture Models |
| **ML Framework** | PyTorch, scikit-learn |
| **Data Processing** | NumPy, Pandas, tiktoken |
| **Web Scraping** | BeautifulSoup4, LangChain RecursiveUrlLoader |
| **Visualization** | Matplotlib |

---

## Project Structure

```
SmartSummarizerRAG/
├── smart-summarizer-rag.ipynb   # Full RAG pipeline (interactive notebook)
├── src/
│   ├── embedding.py             # Text → 1024-dim vectors (Sentence Transformers)
│   ├── clustering.py            # UMAP reduction + GMM clustering
│   ├── retrieval.py             # Cosine similarity search
│   ├── summarization.py         # BART-based abstractive summarization
│   └── visualization.py         # Cluster plots with Matplotlib
├── tests/
│   ├── test_embedding.py
│   ├── test_clustering.py
│   ├── test_retrieval.py
│   └── test_summarization.py
├── models/                      # Local GGUF model files (gitignored)
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- 8GB+ RAM (16GB recommended for 7B models)
- (Optional) NVIDIA GPU with CUDA 11.8+ for accelerated inference

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/SmartSummarizerRAG.git
cd SmartSummarizerRAG

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download a Local LLM

```bash
# Create models directory
mkdir -p models

# Option A: Zephyr-7B (recommended for first run)
wget -P models/ https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf

# Option B: Llama 3.1-8B Instruct
# Download from HuggingFace and place in models/
```

### Run

```bash
# Interactive notebook (recommended)
jupyter notebook smart-summarizer-rag.ipynb

# Or use the modular API
python -c "
from src.embedding import get_embedding
from src.clustering import cluster_embeddings
from src.retrieval import retrieve_similar
from src.summarization import summarize_text

docs = ['Your document text here', 'Another document', 'A third one']
embeddings = [get_embedding(doc) for doc in docs]
labels, reduced = cluster_embeddings(embeddings)
top_indices = retrieve_similar(embeddings[0], embeddings)
summary = summarize_text(docs[0])
print('Cluster labels:', labels)
print('Most similar docs:', top_indices)
print('Summary:', summary)
"

# Run tests
pytest tests/ -v
```

---

## How the RAG Pipeline Works

### Step 1: Document Ingestion
```python
# Scrape and load documents recursively from URLs
loader = RecursiveUrlLoader(url=url, max_depth=20, extractor=bs4_extractor)
docs = loader.load()
```

### Step 2: Token-Aware Chunking
```python
# Split into 1000-token chunks using GPT's tokenizer
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
```

### Step 3: Recursive Embed-Cluster-Summarize
```python
# This is the core innovation — builds a 3-level summary hierarchy
results = recursive_embed_cluster_summarize(texts, level=1, n_levels=3)

# Level 1: 500 chunks → 30 clusters → 30 summaries
# Level 2: 30 summaries → 5 clusters → 5 summaries
# Level 3: 5 summaries → 1 cluster → 1 meta-summary
```

### Step 4: Hierarchical Vector Store
```python
# Store ALL levels — raw chunks + every summary layer
all_texts = original_chunks + L1_summaries + L2_summaries + L3_summaries
vectorstore = Chroma.from_texts(texts=all_texts, embedding=embd)
```

### Step 5: Query with RAG Chain
```python
# LCEL chain: retrieve → format → prompt → LLM → parse
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How to implement LangGraph?")
```

---

## Example Queries & Results

```python
# Detail-level question → retrieves specific chunks
rag_chain.invoke("What is a Self-Query Retriever in LangChain, and what makes it unique?")

# Code-level question → retrieves implementation details
rag_chain.invoke("How to implement LangGraph? Give me a specific code example.")

# Overview question → retrieves high-level summaries
rag_chain.invoke("Provide a detailed summary of all provided documents.")

# Conceptual question → retrieves across multiple levels
rag_chain.invoke("What is LangChain's Expression Language, and how is it used in AI workflows?")
```

---

## Algorithms Deep Dive

### UMAP + GMM Clustering Pipeline

```
Raw Embeddings (1024-dim)
        │
        ▼
   UMAP Reduction ──────── n_neighbors = √n, metric = cosine
        │
        ▼
   2D Embedding Space
        │
        ▼
   GMM Fit (k=1..10) ──── Select k that minimizes BIC
        │
        ▼
   Probability Matrix ──── P(doc ∈ cluster_i) for all i
        │
        ▼
   Threshold Filter ────── Keep assignments where P > 0.1
        │
        ▼
   Cluster Labels ──────── Soft assignments (multi-membership)
```

### Recursive Summarization Hierarchy

```
Level 0 (Leaves):   [chunk₁] [chunk₂] [chunk₃] ... [chunk₅₀₀]
                         \       |       /              |
Level 1 (Clusters):    [summary₁]  [summary₂] ... [summary₃₀]
                            \           |              /
Level 2 (Meta):           [meta₁]  [meta₂] ... [meta₅]
                               \       |       /
Level 3 (Root):              [global_summary]
```

---

## Performance Characteristics

| Metric | Value |
|---|---|
| Embedding Model | BAAI/bge-large-en-v1.5 (1024-dim, MTEB top-tier) |
| Chunk Size | 1000 tokens (tiktoken-calibrated) |
| LLM Context Window | 8192 tokens |
| Quantization | Q4_K_M (4-bit, ~4.5 GB VRAM for 7B model) |
| Clustering | Auto-tuned via BIC (no manual hyperparameters) |
| Multi-processing | Parallel embedding generation across all CPU cores |
| GPU Support | CUDA-accelerated embeddings and inference (optional) |

---

## Extending the System

**Add a REST API:**
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/query")
async def query(question: str):
    return {"answer": rag_chain.invoke(question)}
```

**Swap the LLM:**
```python
# Use any GGUF model — just change the path
llm = ChatLlamaCpp(model_path="models/your-model.gguf", ...)
```

**Add new document sources:**
```python
# PDF, CSV, or any LangChain document loader
from langchain_community.document_loaders import PyPDFLoader
docs = PyPDFLoader("paper.pdf").load()
```

---

## Roadmap

- [ ] FastAPI/Gradio web interface for interactive querying
- [ ] Persistent ChromaDB storage with incremental document updates
- [ ] Support for PDF, CSV, and markdown document loaders
- [ ] Streaming LLM responses
- [ ] Evaluation benchmarks (RAGAS, faithfulness, relevance scores)
- [ ] Docker containerization for one-command deployment

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with</strong> LangChain | ChromaDB | Sentence Transformers | llama.cpp | UMAP | scikit-learn
</p>
