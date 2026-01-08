from brainbox.core.knowledge import Document
from brainbox.core.knowledge.retrievers import KeywordRetriever, VectorRetriever, CompositeRetriever
from brainbox.core.vectorstore.in_memory import InMemoryVectorStore
from brainbox.core.embeddings.ollama import OllamaEmbeddingClient
from brainbox.core.knowledge.indexing import index_documents
from brainbox.pipelines import RAGPipeline
from brainbox.core.llm import OllamaClient
from brainbox.pipelines.nodes.agent_nodes import tool_decision_node # Just to access something if needed, but pipeline encapsulates graph

def run_agent_demo():
    print("Initializing ReAct Agent Pipeline...")
    
    # 1. Setup Knowledge Base (Minimal, as Agent might not use it yet without RetrievalTool)
    # But we pass retriever to build_rag_graph arguments even if unused in current wiring
    embed_client = OllamaEmbeddingClient(model="deepscaler") 
    vector_store = InMemoryVectorStore()
    documents = [Document(id="1", content="Dummy", metadata={})]
    keyword_retriever = KeywordRetriever(documents)
    # Just a dummy retriever to satisfy init
    
    # 2. Setup Pipeline
    client = OllamaClient()
    pipeline = RAGPipeline(retriever=keyword_retriever, client=client)
    
    # 3. Test Multi-Step Logic
    # "Calculate 15 * 6, then subtract 10 from the result."
    # Expect:
    # 1. Tool(calculator, "15 * 6") -> 90
    # 2. Tool(calculator, "90 - 10") -> 80
    # 3. Final Answer: 80
    
    question = "Calculate 15 * 6, then subtract 10 from the result."
    print(f"\n[Scenario 1] User asks: '{question}'")
    print("-" * 50)
    
    result = pipeline.run(question)
    
    print(f"DECISION: {result.final_decision}")
    
    # Inspect History if available in result (RunRecord doesn't show history directly?)
    # We can inspect the returned answer.
    # Agent "Final Answer" is not currently returned in 'answer' field of RunRecord?
    # RunRecord logic:
    # answer = "Tool Output: ..." OR "response.text"
    # In agent_nodes.py, on "ANSWER_READY", we didn't populate tool_output or response.
    # We need to fix RAGPipeline.run mapping to capture Agent Answer!
    
    # Let's see what we get.
    print(f"ANSWER:   {result.answer}")
    print("-" * 50)

if __name__ == "__main__":
    run_agent_demo()
