import os
import sys
import argparse
from typing import List
from rag import Chunker, EmbeddingModel, VectorStore, Generator

def ingest_directory(directory_path: str, vector_store: VectorStore, embed_model: EmbeddingModel, chunker: Chunker):
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Find all text and markdown files
    files_to_process = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".txt", ".md", ".json")):
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print(f"No text/markdown files found in {directory_path}.")
        return

    print(f"Found {len(files_to_process)} files to ingest. Processing...")
    
    all_chunks = []
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            source_name = os.path.basename(file_path)
            chunks = chunker.split_text(content, source_name)
            all_chunks.extend(chunks)
            print(f"  Processed {source_name}: split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"  Error reading {file_path}: {e}")

    if not all_chunks:
        print("No content could be extracted.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    texts = [chunk["text"] for chunk in all_chunks]
    embed_model.fit(texts)
    embeddings = embed_model.encode(texts)

    vector_store.add_documents(all_chunks, embeddings)
    vector_store.save()
    print("Ingestion completed successfully!")

def run_query(query_text: str, vector_store: VectorStore, embed_model: EmbeddingModel, generator: Generator, top_k: int = 3):
    if not vector_store.load():
        print("Error: No vector store index found. Please run ingestion first.")
        return

    # Fit TF-IDF model on the loaded documents
    texts = [doc["text"] for doc in vector_store.documents]
    embed_model.fit(texts)

    print(f"Querying: '{query_text}'")
    query_embedding = embed_model.encode([query_text])[0]
    
    # Retrieve top K matched documents
    results = vector_store.search(query_embedding, top_k=top_k)
    
    if not results:
        print("No matching context found.")
        return

    print("\n--- Retrieved Contexts ---")
    retrieved_contexts = []
    for idx, (doc, similarity) in enumerate(results, 1):
        print(f"[{idx}] Similarity: {similarity:.4f} (Source: {doc['metadata']['source']})")
        print(f"    Text: {doc['text'][:150]}...")
        retrieved_contexts.append(doc)
    print("--------------------------\n")

    # Generate answer
    print("Generating response...")
    response = generator.generate(query_text, retrieved_contexts)
    print("\n--- Answer ---")
    print(response)
    print("--------------\n")

def main():
    parser = argparse.ArgumentParser(description="Simple Local RAG Implementation")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory of documents")
    ingest_parser.add_argument("dir", type=str, help="Directory containing documents (.txt, .md)")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("text", type=str, help="Query text / question")
    query_parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")

    args = parser.parse_args()

    # Initialize RAG components
    chunker = Chunker(chunk_size=150, chunk_overlap=30)
    embed_model = EmbeddingModel()
    vector_store = VectorStore()
    generator = Generator()

    if args.command == "ingest":
        ingest_directory(args.dir, vector_store, embed_model, chunker)
    elif args.command == "query":
        run_query(args.text, vector_store, embed_model, generator, top_k=args.top_k)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
