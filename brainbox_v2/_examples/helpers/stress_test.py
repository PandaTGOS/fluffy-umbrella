import sys
import os
import json
import time

# Ensure we can import brainbox
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brainbox.core.knowledge.documents import Document
from brainbox.core.vectorstore.chroma import ChromaVectorStore
from brainbox.core.knowledge.indexing.inverted_index import InvertedIndex
from brainbox.core.security.rbac import RBACManager, User, Role
from brainbox.core.routing.prefix_router import PrefixRouter
from brainbox.core.knowledge.graph.chunk_graph import ChunkGraph
from brainbox.pipelines.secure_enterprise_rag import SecureEnterpriseRAG

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")

def parse_frontmatter(content: str) -> dict:
    # Very simple frontmatter parser assuming --- json --- format
    try:
        parts = content.split("---", 2)
        if len(parts) >= 3:
            metadata = json.loads(parts[1])
            body = parts[2].strip()
            return metadata, body
    except Exception as e:
        print(f"Error parsing frontmatter: {e}")
    return {}, content

def load_data():
    if not os.path.exists(DATA_DIR):
        print("Data directory not found. Please run generate_large_data.py first.")
        sys.exit(1)
        
    documents = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".md")]
    print(f"Loading {len(files)} documents from {DATA_DIR}...")
    
    for filename in files:
        with open(os.path.join(DATA_DIR, filename), "r") as f:
            content = f.read()
            metadata, body = parse_frontmatter(content)
            
            # If id missing in metadata, fallback to filename
            doc_id = metadata.get("id", filename)
            
            documents.append({
                "id": doc_id,
                "content": body,
                "metadata": metadata
            })
    return documents

def run_stress_test():
    raw_data = load_data()
    docs = [Document(id=d["id"], content=d["content"], metadata=d["metadata"]) for d in raw_data]
    
    print(f"Loaded {len(docs)} documents.")

    # 1. Initialize Components
    print("Initializing components...")
    vector_store = ChromaVectorStore(collection_name="stress_test", persist_directory="./chroma_stress_db")
    inverted_index = InvertedIndex()
    rbac = RBACManager()
    router = PrefixRouter()
    graph = ChunkGraph()

    # 2. Ingestion Stress Test
    print("\n--- Starting Ingestion Stress Test ---")
    start_time = time.time()
    
    # Batch ingestion if possible, but our interfaces are simple
    # Vector store supports batch
    vector_store.add(docs)
    
    # Inverted Index and Graph are iterative in this simple impl
    for doc in docs:
        inverted_index.add(doc)
    
    graph.build(docs) # Graph build supports list
    
    end_time = time.time()
    print(f"Ingestion completed in {end_time - start_time:.2f} seconds.")
    print(f"Average time per doc: {(end_time - start_time) / len(docs) * 1000:.2f} ms")

    # 3. Pipeline Setup
    pipeline = SecureEnterpriseRAG(vector_store, inverted_index, rbac, router, graph)
    
    # 4. Retrieval Stress Test
    print("\n--- Starting Retrieval Stress Test ---")
    queries = ["finance report", "engineering design", "hr policies", "security breach", "sales targets"] * 20
    user = User(id="u1", roles=[Role.ADMIN]) # Admin sees everything
    
    start_time = time.time()
    for q in queries:
        pipeline.run(q, user)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / len(queries)
    print(f"Executed {len(queries)} queries in {total_time:.2f} seconds.")
    print(f"Average Query Latency: {avg_latency * 1000:.2f} ms")

    # 5. Peak Memory / Cleanup (Optional - simplistic check)
    if hasattr(vector_store, 'use_mock') and vector_store.use_mock:
        print(f"Mock Vector Store Size: {len(vector_store.mock_store)}")
    
    print("\nStress Test Completed.")

if __name__ == "__main__":
    run_stress_test()
