import os
import json
from flask import Flask, jsonify, request, send_from_directory
from rag import Chunker, EmbeddingModel, VectorStore, Generator

app = Flask(__name__, static_folder='static')

# Initialize RAG components
chunker = Chunker(chunk_size=150, chunk_overlap=30)
embed_model = EmbeddingModel()
vector_store = VectorStore()
generator = Generator()

# Load the vector store and fit the embedding model at startup if index exists
def init_rag():
    try:
        if vector_store.load():
            if vector_store.documents:
                texts = [doc["text"] for doc in vector_store.documents]
                embed_model.fit(texts)
                print("RAG components successfully initialized and loaded from disk.")
    except Exception as e:
        print(f"Error loading vector store at startup: {e}")

init_rag()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/status', methods=['GET'])
def get_status():
    exists = os.path.exists(vector_store.filepath)
    chunk_count = len(vector_store.documents) if exists else 0
    return jsonify({
        "exists": exists,
        "chunk_count": chunk_count,
        "filepath": vector_store.filepath
    })

@app.route('/api/ingest', methods=['POST'])
def ingest():
    data = request.get_json() or {}
    directory_path = data.get('directory', './data')

    if not os.path.exists(directory_path):
        return jsonify({"error": f"Directory '{directory_path}' does not exist."}), 400

    # Find all text, markdown, and json files
    files_to_process = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".txt", ".md", ".json")):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        return jsonify({"error": f"No text/markdown/json files found in {directory_path}."}), 400

    all_chunks = []
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            source_name = os.path.basename(file_path)
            chunks = chunker.split_text(content, source_name)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not all_chunks:
        return jsonify({"error": "No content could be extracted from documents."}), 400

    # Fit TF-IDF model and encode documents
    try:
        texts = [chunk["text"] for chunk in all_chunks]
        embed_model.fit(texts)
        embeddings = embed_model.encode(texts)

        # Clear existing documents and add new ones
        vector_store.embeddings = embeddings
        vector_store.documents = all_chunks
        vector_store.save()
        
        return jsonify({
            "message": f"Successfully ingested {len(files_to_process)} files into {len(all_chunks)} chunks.",
            "chunk_count": len(all_chunks)
        })
    except Exception as e:
        return jsonify({"error": f"Ingestion error: {str(e)}"}), 500

@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json() or {}
    query_text = data.get('query', '')
    top_k = int(data.get('top_k', 3))

    if not query_text:
        return jsonify({"error": "Query text cannot be empty."}), 400

    # Check if vector store is loaded
    if not vector_store.documents:
        # Try loading
        if not vector_store.load():
            return jsonify({"error": "No vector store index found. Please run ingestion first."}), 400
        
        # Fit embeddings if just loaded
        texts = [doc["text"] for doc in vector_store.documents]
        embed_model.fit(texts)

    try:
        # Encode query
        query_embedding = embed_model.encode_query(query_text)
        
        # Search vector database
        results = vector_store.search(query_embedding, top_k=top_k)
        
        # Generate response using LLM generator
        retrieved_contexts = [doc for doc, similarity in results]
        response_text = generator.generate(query_text, retrieved_contexts)
        
        # Get query token matches and scores
        token_matches = []
        try:
            token_matches = embed_model.get_query_terms_with_scores(query_text)
        except Exception as e:
            print(f"Error extracting query token matches: {e}")
        
        # Format contexts to return to client
        contexts_to_return = []
        for doc, similarity in results:
            contexts_to_return.append({
                "text": doc["text"],
                "source": doc["metadata"]["source"],
                "similarity": float(similarity)
            })
            
        return jsonify({
            "answer": response_text,
            "contexts": contexts_to_return,
            "token_matches": token_matches
        })
    except Exception as e:
        return jsonify({"error": f"Query processing error: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    directory_path = request.args.get('directory', './data')
    if not os.path.exists(directory_path):
        return jsonify([])
    
    files_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".txt", ".md", ".json")):
                rel_path = os.path.relpath(os.path.join(root, file), directory_path)
                full_path = os.path.join(root, file)
                files_list.append({
                    "name": file,
                    "rel_path": rel_path.replace('\\', '/'),
                    "size": os.path.getsize(full_path)
                })
    return jsonify(files_list)

@app.route('/api/documents/read', methods=['GET'])
def read_document():
    directory_path = request.args.get('directory', './data')
    rel_path = request.args.get('path', '')
    if not rel_path:
        return jsonify({"error": "Path parameter is required."}), 400
    
    # Safe path resolution
    safe_path = os.path.abspath(os.path.join(directory_path, rel_path))
    if not safe_path.startswith(os.path.abspath(directory_path)):
        return jsonify({"error": "Access denied."}), 403
        
    if not os.path.exists(safe_path):
        return jsonify({"error": "File not found."}), 404
        
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/save', methods=['POST'])
def save_document():
    data = request.get_json() or {}
    directory_path = data.get('directory', './data')
    rel_path = data.get('path', '')
    content = data.get('content', '')
    
    if not rel_path:
        return jsonify({"error": "Path parameter is required."}), 400
        
    safe_path = os.path.abspath(os.path.join(directory_path, rel_path))
    if not safe_path.startswith(os.path.abspath(directory_path)):
        return jsonify({"error": "Access denied."}), 403
        
    try:
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({"message": f"Successfully saved {rel_path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/delete', methods=['POST'])
def delete_document():
    data = request.get_json() or {}
    directory_path = data.get('directory', './data')
    rel_path = data.get('path', '')
    
    if not rel_path:
        return jsonify({"error": "Path parameter is required."}), 400
        
    safe_path = os.path.abspath(os.path.join(directory_path, rel_path))
    if not safe_path.startswith(os.path.abspath(directory_path)):
        return jsonify({"error": "Access denied."}), 403
        
    if not os.path.exists(safe_path):
        return jsonify({"error": "File not found."}), 404
        
    try:
        os.remove(safe_path)
        return jsonify({"message": f"Successfully deleted {rel_path}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask Dev Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
