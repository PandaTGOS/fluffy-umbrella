import os
import sys
sys.path.append(os.getcwd())

from brainbox.core.llm import OllamaClient
from brainbox.core.embeddings.ollama import OllamaEmbeddingClient
from brainbox.core.vectorstore.in_memory import InMemoryVectorStore
from brainbox.core.knowledge.chunking.recursive import RecursiveChunker
from brainbox.core.knowledge.ingestion.knowledge_base import DirectoryKnowledgeBase
from brainbox.pipelines.rag_pipeline import RAGPipeline

def main():
    print("Running Ingestion Demo")
    
    # 1. Setup Infrastructure
    # Use RecursiveChunker for structure-aware splitting
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64) 
    vector_store = InMemoryVectorStore()

    # STRICTLY REAL CLIENTS
    print("🔌 Connecting to Ollama...")
    embeddings = OllamaEmbeddingClient()
    llm = OllamaClient()

    # 2. Create Knowledge Base from Directory
    print("\n📂 Scanning 'data/demo_docs'...")
    kb = DirectoryKnowledgeBase.from_path(
        path="data/demo_docs",
        chunker=chunker,
        embedding_client=embeddings,
        vector_store=vector_store
    )

    # 3. Create Pipeline
    pipeline = RAGPipeline(
        retriever=kb.as_retriever(),
        client=llm
    )

    # 4. Ask Questions
    questions = [
        "What is the capital of France?", 
        "What are the engineering principles?",
        "What was agreed upon in the meeting notes?"
    ]

    for q in questions:
        print(f"\n❓ Question: {q}")
        result = pipeline.run(q)
        print(f"✅ Answer: {result.answer}")
        
        if result.raw_result.get("documents"):
            top_doc = result.raw_result["documents"][0]
            print(f"   (Source: {top_doc['metadata']['source']})")
            print(f"   (Snippet: {top_doc['content'][:50]}...)")

if __name__ == "__main__":
    main()
