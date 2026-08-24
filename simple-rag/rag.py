import os
import json
import ssl
import numpy as np
from typing import List, Dict, Any, Tuple

# Bypass SSL certificate verification (common issue on Windows/internal networks)
ssl._create_default_https_context = ssl._create_unverified_context

# Monkeypatch requests and httpx to globally disable SSL verification
try:
    import requests
    import httpx
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Patch requests
    original_request = requests.Session.request
    def unverified_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)
    requests.Session.request = unverified_request

    # Patch httpx
    original_httpx_init = httpx.Client.__init__
    def unverified_httpx_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_httpx_init(self, *args, **kwargs)
    httpx.Client.__init__ = unverified_httpx_init
except Exception:
    pass

# TF-IDF Embedding model in pure Python (no external downloads needed, bypasses Zscaler/firewall blocks)
import math
import re
from collections import Counter
import google.generativeai as genai

class Chunker:
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source_name: str) -> List[Dict[str, Any]]:
        """Splits text into chunks of words with overlapping sliding window."""
        words = text.split()
        if not words:
            return []
            
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_name,
                    "word_count": len(chunk_words),
                    "start_word_idx": start
                }
            })
            
            start += self.chunk_size - self.chunk_overlap
            
        return chunks

class EmbeddingModel:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, documents: List[str]):
        """Fits the vocabulary and IDF on the corpus of document chunks."""
        # Simple tokenization: lowercase and alphanumeric words
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        
        # Build vocabulary
        vocab = set()
        for doc in tokenized_docs:
            vocab.update(doc)
        self.vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
        
        # Calculate IDF
        num_docs = len(documents)
        self.idf = {}
        for word in self.vocab:
            # Count how many docs contain the word
            doc_count = sum(1 for doc in tokenized_docs if word in doc)
            # Smooth IDF
            self.idf[word] = math.log((1 + num_docs) / (1 + doc_count)) + 1

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def encode(self, texts: List[str]) -> np.ndarray:
        """Transforms texts into normalized TF-IDF vectors."""
        if not self.vocab:
            # If not fitted yet, just return zeros or simple representation
            return np.zeros((len(texts), 1))
            
        vectors = []
        for text in texts:
            tokens = self._tokenize(text)
            counts = Counter(tokens)
            
            vec = np.zeros(len(self.vocab))
            for word, count in counts.items():
                if word in self.vocab:
                    tf = count / len(tokens) if tokens else 0
                    vec[self.vocab[word]] = tf * self.idf[word]
            
            # Normalize vector to unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
                
            vectors.append(vec)
            
        return np.array(vectors)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query text."""
        return self.encode([query])[0]

    def get_query_terms_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """Returns the tokenized words in the query that match the vocabulary, along with their IDF weights."""
        tokens = self._tokenize(query)
        counts = Counter(tokens)
        
        results = []
        for word in sorted(set(tokens)):
            if word in self.vocab:
                idf_val = self.idf[word]
                tf = counts[word] / len(tokens) if tokens else 0
                results.append({
                    "word": word,
                    "tf": tf,
                    "idf": idf_val,
                    "tfidf": tf * idf_val
                })
        # Sort by tfidf descending
        results.sort(key=lambda x: x["tfidf"], reverse=True)
        return results


class VectorStore:
    def __init__(self, filepath: str = "vector_store.npz"):
        self.filepath = filepath
        self.embeddings: np.ndarray = np.empty((0, 0))
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Adds documents and their embeddings to the store."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")
            
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
        self.documents.extend(documents)

    def save(self):
        """Saves embeddings and document metadata to disk."""
        np.savez(
            self.filepath, 
            embeddings=self.embeddings, 
            documents=json.dumps(self.documents)
        )
        print(f"Saved {len(self.documents)} chunks to {self.filepath}")

    def load(self) -> bool:
        """Loads embeddings and document metadata from disk if exists."""
        if not os.path.exists(self.filepath):
            return False
            
        data = np.load(self.filepath, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.documents = json.loads(str(data["documents"]))
        print(f"Loaded {len(self.documents)} chunks from {self.filepath}")
        return True

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Performs cosine similarity search against stored embeddings."""
        if self.embeddings.size == 0:
            return []
            
        # Cosine similarity calculation: (A . B) / (||A|| * ||B||)
        # Normalize embeddings to unit length for easier dot product computation
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        norm_embeddings = self.embeddings / norms
        
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            query_norm = 1e-9
        norm_query = query_embedding / query_norm
        
        similarities = np.dot(norm_embeddings, norm_query)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
            
        return results


class Generator:
    def __init__(self):
        # Configure Gemini API using environment variable or fallback to hardcoded API key
        api_key = os.environ.get("GEMINI_API_KEY") or ""
        if api_key:
            genai.configure(api_key=api_key)
            # Use gemini-3.5-flash as default recommended model
            self.model = genai.GenerativeModel("gemini-3.5-flash")
            self.use_api = True
        else:
            self.model = None
            self.use_api = False

    def generate(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generates response using retrieved context and prompt template."""
        context_str = "\n\n".join([
            f"--- Context Segment (Source: {c['metadata']['source']}) ---\n{c['text']}"
            for c in contexts
        ])
        
        prompt = f"""You are the official Acme Corp HR Assistant & Employee Helpline. Answer the query professionally, accurately, and empathetically using ONLY the provided context below.
If the answer cannot be found in the context, say "I'm sorry, I cannot find that information in our HR handbook or helpline documents. Please contact the HR team directly at hr@acme-corp-example.com."

Context:
{context_str}

Query: {query}

Answer:"""

        if self.use_api:
            # Try configured model, fallback to alternative models if rate-limited or unavailable
            models_to_try = [
                os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                "gemini-3.1-flash-lite",
                "gemini-2.0-flash"
            ]
            # De-duplicate while preserving order
            models_to_try = list(dict.fromkeys(m for m in models_to_try if m))
            
            last_err = None
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    last_err = e
                    print(f"Failed to generate with {model_name}: {e}")
                    continue
            return f"[Error calling Gemini API: {str(last_err)}]\n\nPrompt constructed:\n{prompt}"
        else:
            # Local Mock fallback description
            fallback = "[Note: GEMINI_API_KEY not set in environment. Running in mock/offline mode]\n\n"
            fallback += f"Constructed Context-Augmented Prompt:\n{'-'*50}\n{prompt}\n{'-'*50}"
            return fallback
