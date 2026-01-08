from brainbox.core.knowledge import Document
from brainbox.core.knowledge.retrievers import KeywordRetriever
from brainbox.pipelines.rag_graph import build_rag_graph
from brainbox.core.llm import OllamaClient
from brainbox.core.state.rag_state import RAGState
from brainbox.core.knowledge.retrievers.composite import CompositeRetriever
from brainbox.core.knowledge.rerankers.llm_reranker import LLMReranker

def verify_multi_agent():
    print("Verifying Multi-Agent Architecture (Phase F Complete)...")
    
    # Setup
    client = OllamaClient()
    
    # Underlying Retriever (Knowledge Base)
    base_retriever = KeywordRetriever([
        Document(id="1", content="Paris is the capital of France.", metadata={"source": "manual"}),
        Document(id="2", content="Berlin is the capital of Germany.", metadata={"source": "manual"})
    ])
    
    # Phase F: Composite + Reranker
    reranker = LLMReranker(client)
    composite_retriever = CompositeRetriever(retrievers=[base_retriever], reranker=reranker)
    
    # Build graph with CompositeRetriever
    graph = build_rag_graph(composite_retriever, client)
    
    # Scenario 2: RAG
    # ... (Keep existing S2 but print less)
    
    # Scenario 3: General Chat (AnswerAgent)
    print("\n[Scenario 3] General Chat (Expect: AnswerAgent)")
    # Query with no keywords matching mock docs ("Paris", "Berlin")
    state3 = RAGState(question="Tell me a joke about AI.")
    
    print("Invoking Graph...")
    result3 = graph.invoke(state3)
    
    # Check if AnswerAgent was used (inferred from response content or logs if we had them)
    # We can check documents count (should be 0 or low relevance)
    docs3 = result3.get("documents", [])
    print(f"Retrieved Documents: {len(docs3)}")
    print(f"Signals: {result3.get('retrieval_signals')}")
    
    response3 = result3.get('response')
    print(f"Response: {response3.text if response3 else 'None'}")
    
    # Check Agent History if available or inferred
    # AnswerAgent leaves no history trace in `agent_history` unless we log it? 
    # But result contains response.
    
if __name__ == "__main__":
    verify_multi_agent()
