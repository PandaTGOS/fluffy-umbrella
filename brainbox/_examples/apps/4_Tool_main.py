from brainbox.core.knowledge import Document
from brainbox.core.knowledge.retrievers import KeywordRetriever, VectorRetriever, CompositeRetriever
from brainbox.core.vectorstore.in_memory import InMemoryVectorStore
from brainbox.core.embeddings.ollama import OllamaEmbeddingClient
from brainbox.core.knowledge.indexing import index_documents
from brainbox.pipelines import RAGPipeline
from brainbox.core.llm import OllamaClient

def run_tool_demo():
    print("Initializing RAG Pipeline with Tool Support + Composite Retrieval...")
    
    # 1. Setup Knowledge Base
    documents = [
        Document(
            id="fr_cap",
            content="The capital of France is Paris.",
            metadata={"source": "geography_book"},
        ),
         Document(
            id="py_fact",
            content="Python is a programming language.",
            metadata={"source": "cs_book"},
        )
    ]
    
    # 2. Setup Retrievers
    print("  - Indexing Vector Store...")
    # Using 'deepscaler' for embeddings since we know it's available, 
    # though usually dedicated embed models are better.
    embed_client = OllamaEmbeddingClient(model="deepscaler")
    vector_store = InMemoryVectorStore()
    
    # helper to index (embed -> add)
    index_documents(documents, embed_client, vector_store)
    
    vector_retriever = VectorRetriever(embed_client, vector_store)
    keyword_retriever = KeywordRetriever(documents)
    
    composite_retriever = CompositeRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        overfetch=10
    )
    
    # 3. Setup Pipeline
    client = OllamaClient()
    pipeline = RAGPipeline(retriever=composite_retriever, client=client)
    
    # 4. Test Math Tool (Should rely on tool_decision -> calculator)
    question = "50 + 25"
    print(f"\n[Scenario 1] User asks: '{question}'")
    print("-" * 50)
    
    result = pipeline.run(question)
    
    print(f"DECISION: {result.final_decision}")
    print(f"ANSWER:   {result.answer}")
    print("-" * 50)
    
    # 5. Test RAG Path (Should rely on Composite -> retrieve -> llm)
    question_rag = "What is the capital of France?"
    print(f"\n[Scenario 2] User asks: '{question_rag}'")
    print("-" * 50)
    
    result_rag = pipeline.run(question_rag)
    
    print(f"DECISION: {result_rag.final_decision}")
    print(f"ANSWER:   {result_rag.answer}")
    print(f"RETRIEVER: {result_rag.retriever_type}")
    if result_rag.attempts:
        print(f"CONFIDENCE: {result_rag.confidence}")
        # Inspect Provenance from the first retrieved doc in the successful attempt context?
        # The RunRecord doesn't expose retrieved docs directly, but the logs inside pipeline might if verbose.
        # But we can trust verify_composite logic.
    print("-" * 50)
    
    # 3. Test Python Tool
    # Note: Our heuristic looks for "python" or "code".
    # Let's use "code" keyword but clean format
    question_py = "code x = 10 * 5\nprint(x)"
    print(f"\n[Scenario 3] User asks: '{question_py}'")
    print("-" * 50)
    
    result_py = pipeline.run(question_py)
    
    print(f"DECISION: {result_py.final_decision}")
    print(f"ANSWER:   {result_py.answer}")
    print("-" * 50)

if __name__ == "__main__":
    run_tool_demo()
