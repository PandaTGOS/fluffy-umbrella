from brainbox.core.llm import OllamaClient
from brainbox.core.knowledge.ingestion.knowledge_base import DirectoryKnowledgeBase
from brainbox.core.knowledge.chunking.recursive import RecursiveChunker
from brainbox.core.embeddings.ollama import OllamaEmbeddingClient
from brainbox.core.vectorstore.in_memory import InMemoryVectorStore
from brainbox.pipelines.agent_pipeline import AgentPipeline

def run_prime_multi_agent_orchestration():
    print("🚀 Initializing Multi-Agent Orchestration System...")
    
    # 1. LLM Client
    client = OllamaClient(default_model="deepscaler")

    # 2. Knowledge Base (Automated Directory Ingestion)
    # Using the DirectoryKnowledgeBase to load real documents from disk
    kb = DirectoryKnowledgeBase.from_path(
        path="data/demo_docs",
        chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
        embedding_client=OllamaEmbeddingClient(),
        vector_store=InMemoryVectorStore()
    )
    retriever = kb.as_retriever()

    # 3. Agent Pipeline (The Orchestrator)
    # This orchestrator manages a pool of specialized agents:
    # - RetrievalAgent (Researcher)
    # - ToolAgent (Execution Specialist)
    # - AnswerAgent (Synthesis expert)
    # - CriticAgent (Quality Control)
    orchestrator = AgentPipeline(retriever=retriever, client=client)

    # 4. Complex Agentic Task
    complex_task = (
        "Briefly explain why recursive chunking is good for RAG based on our internal docs, "
        "and then calculate how many 500-character chunks would be created "
        "from a document of 25,000 characters if there is no overlap."
    )
    
    print(f"\n📝 Orchestrating Task: {complex_task}")
    print("\n--- Multi-Agent Flow Started ---")
    
    result = orchestrator.run(complex_task)
    
    print("\n--- Final Consolidated Output ---")
    print(f"Status: {result.final_decision}")
    print(f"Response:\n{result.answer}")

if __name__ == "__main__":
    run_prime_multi_agent_orchestration()
