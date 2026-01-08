from brainbox.core.knowledge.documents import Document
from brainbox.core.knowledge.chunking.fixed import FixedChunker
from brainbox.core.knowledge.indexing.indexer import index_documents
from brainbox.core.vectorstore.in_memory import InMemoryVectorStore
from brainbox.core.llm import OllamaClient # Assuming OllamaClient implements EmbeddingClient too? 
# Wait, OllamaClient in core/llm/llm_client.py. Does it have embed()?
# core/embeddings/base.py defines EmbeddingClient.
# I might need to check if OllamaClient implements it or if there is a separate OllamaEmbeddingClient.
# Assuming OllamaClient has .embed or checking imports.
# Let's check imports in apps/10...
# core/embeddings/ollama.py ?
# I'll check directory.

from brainbox.core.embeddings.ollama import OllamaEmbeddingClient

def verify_indexing():
    print("Verifying Indexing Pipeline (Phase I)...")
    
    # 1. Setup Components
    chunker = FixedChunker(size=50, overlap=10) # Small chunks for demo
    
    # Embedding Client
    # Need to check if OllamaEmbeddingClient exists.
    # Assuming standard pattern.
    emb_client = OllamaEmbeddingClient()
    
    # Vector Store
    vector_store = InMemoryVectorStore()
    
    # 2. Prepare Documents
    long_text = "The quick brown fox jumps over the lazy dog. " * 5
    docs = [
        Document(id="doc1", content=long_text, metadata={"source": "test"})
    ]
    
    # 3. Run Indexing Pipeline
    print("\n[Indexer] Running index_documents...")
    index_documents(docs, chunker, emb_client, vector_store)
    
    # 4. Verify Storage
    # InMemoryVectorStore stores in .vectors? or we can search.
    # We can try to search "fox".
    print("\n[VectorStore] Searching for 'fox'...")
    query_vec = emb_client.embed(["fox"])[0]
    results = vector_store.search(query_vec, k=3)
    
    print(f"Found {len(results)} results.")
    for res in results:
        print(f"Content: {res['metadata']['content']} (Score: {res['score']})")
        
if __name__ == "__main__":
    verify_indexing()
