from brainbox.core.knowledge import Document
from brainbox.core.knowledge.retrievers import (
    KeywordRetriever,
    VectorRetriever,
    CompositeRetriever
)
from brainbox.core.knowledge.rerankers.llm_reranker import LLMReranker
from brainbox.core.embeddings.ollama import OllamaEmbeddingClient
from brainbox.core.vectorstore.in_memory import InMemoryVectorStore
from brainbox.core.knowledge.chunking.fixed import FixedChunker
from brainbox.core.knowledge.indexing.indexer import index_documents
from brainbox.core.llm import OllamaClient
from brainbox.pipelines.rag_pipeline import RAGPipeline

def main():
    print("🌟 Running Golden RAG App...")

    # -----------------------------
    # 0. Core Components
    # -----------------------------
    llm = OllamaClient()  # Initialize LLM first for Reranker/Pipeline

    # -----------------------------
    # 1. Knowledge Base
    # -----------------------------
    documents = [
        Document(
            id="france_capital",
            content="The capital of France is Paris.",
            metadata={"source": "wikipedia"}
        )
    ]

    # -----------------------------
    # 2. Vector Index
    # -----------------------------
    chunker = FixedChunker(size=256, overlap=32)
    embeddings = OllamaEmbeddingClient()
    vector_store = InMemoryVectorStore()

    # Indexing Pipeline
    index_documents(
        documents=documents,
        chunker=chunker,
        embedding_client=embeddings,
        vector_store=vector_store
    )

    # -----------------------------
    # 3. Retrievers
    # -----------------------------
    keyword = KeywordRetriever(documents)
    
    # Vector Retriever needs to search the store we just indexed
    vector = VectorRetriever(embeddings, vector_store)
    
    # Reranker (Optional but Recommended)
    reranker = LLMReranker(client=llm)

    # Hybrid Retrieval with Observability
    retriever = CompositeRetriever(
        retrievers=[keyword, vector],
        reranker=reranker,
        overfetch=10
    )

    # -----------------------------
    # 4. Pipeline
    # -----------------------------
    pipeline = RAGPipeline(
        retriever=retriever,
        client=llm
    )

    # -----------------------------
    # 5. Run
    # -----------------------------
    question = "What is the capital of France?"
    result = pipeline.run(question)

    print("\nQUESTION:", question)
    print("ANSWER:", result.answer)
    print("DECISION:", result.final_decision)
    print("CONFIDENCE:", result.confidence)
    
    # Observability Check
    if result.raw_result.get("retrieval_signals"):
        print("SIGNALS:", result.raw_result["retrieval_signals"])

if __name__ == "__main__":
    main()
