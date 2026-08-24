# Simple RAG System

This is a simple, lightweight Retrieval-Augmented Generation (RAG) implementation in Python.

## Standard RAG Architecture vs. Our Implementation

Retrieval-Augmented Generation (RAG) typically relies on complex, heavy enterprise technologies. Below is an overview of standard RAG components and how this codebase replaces them with lightweight, native Python implementations:

### 1. Document Chunking
* **Standard Technology**: Libraries like LangChain (`RecursiveCharacterTextSplitter`) or LlamaIndex.
* **Our Replacement**: The [Chunker](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py#L39) class in [rag.py](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py). It uses a simple word-based sliding window (`split_text`) that tokenizes by splitting whitespace, ensuring zero external overhead.

### 2. Embedding Model
* **Standard Technology**: Heavy deep learning models (e.g., Hugging Face's `sentence-transformers`) or hosted API-based models (e.g., OpenAI's `text-embedding-3-small`).
* **Our Replacement**: The [EmbeddingModel](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py#L70) class in [rag.py](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py). It implements a pure Python TF-IDF vectorizer (`fit`, `encode`, `encode_query`) using `math`, `re`, and `collections.Counter` to construct normalized frequency-based vector representations without downloading models or needing internet/GPU access.

### 3. Vector Database
* **Standard Technology**: Dedicated databases like Pinecone, ChromaDB, Weaviate, Milvus, or Qdrant.
* **Our Replacement**: The [VectorStore](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py#L149) class in [rag.py](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py). It utilizes in-memory `numpy` arrays for raw embeddings and standard JSON serialization for metadata. Similarity search is performed using matrix operations (`np.dot` and `np.linalg.norm`) for cosine similarity, and state is persisted to disk via a simple compressed NumPy file (`vector_store.npz`).

### 4. LLM Generation & Orchestration
* **Standard Technology**: Orchestrators like LangChain/LlamaIndex pipelines connected to heavy API integrations.
* **Our Replacement**: The [Generator](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py#L213) class in [rag.py](file:///c:/Users/atvikass/.gemini/antigravity-ide/scratch/simple-rag/rag.py). It directly wraps the official `google-generativeai` SDK (targeting `gemini-3.5-flash` with fallbacks) and uses basic Python string formatting to construct context-augmented prompts without extra orchestration layers.

---

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Gemini API Key (Optional)**:
   To run LLM generation, set your API key in the environment:
   - Windows (PowerShell):
     ```powershell
     $env:GEMINI_API_KEY="your-api-key"
     ```
   - Linux/macOS:
     ```bash
     export GEMINI_API_KEY="your-api-key"
     ```

## Usage

### 1. Ingest Documents
Ingest all text files in a directory (like the sample `./data/` directory):
```bash
python main.py ingest ./data
```

### 2. Query
Query the system:
```bash
python main.py query "Who is the CEO of Acme Corp?"
```
Or query offline to see the constructed prompt:
```bash
python main.py query "Does AgentForge run offline?"
```
