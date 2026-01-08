from brainbox.core.knowledge import Document
from brainbox.core.knowledge.retrievers import KeywordRetriever
from brainbox.pipelines import RAGPipeline
from brainbox.core.llm import OllamaClient
from brainbox.core.embeddings.ollama import OllamaEmbeddingClient

def run_router_demo():
    print("Initializing Router-Graph Pipeline...")
    
    # Minimal Init
    embed_client = OllamaEmbeddingClient(model="deepscaler")
    documents = [Document(id="1", content="Dummy", metadata={})]
    keyword_retriever = KeywordRetriever(documents)
    client = OllamaClient()
    pipeline = RAGPipeline(retriever=keyword_retriever, client=client)
    
    # Scenario 1: Deterministic Math
    q1 = "50 * 10 + 5"
    print(f"\n[Scenario 1] User asks: '{q1}' (Expect: Router -> Calculator -> END)")
    print("-" * 50)
    res1 = pipeline.run(q1)
    # Since we end with None, result object might be sparse?
    # RAGPipeline.run returns RunRecord.
    # It constructs RunRecord from final state.
    # If final_decision is None, answer might be empty unless we extract from tool_result.
    # I verified earlier `pipeline.run` logic checks `final_state.get("response")`.
    # Deterministic Calculator does not set "response".
    # So `result.answer` might be None.
    # We should inspect `result.final_state["agent_history"][-1]["tool_result"]`.
    # But RunRecord doesn't expose final_state directly?
    # Wait, `pipeline.run` implementation:
    # return RunRecord(..., answer=answer, ...)
    # If answer is None, it prints None.
    # I should check the logs for "[AGENT] Executing..." and presence of LLM calls.
    print(f"Final Decision: {res1.final_decision}") 
    
    
    # Scenario 2: Agent Question
    q2 = "Translate 'Hello' to Spanish"
    # Router "translate" regex? No. Default -> LLM.
    print(f"\n[Scenario 2] User asks: '{q2}' (Expect: Router -> LLM)")
    print("-" * 50)
    res2 = pipeline.run(q2)
    print(f"Answer: {res2.answer}")

if __name__ == "__main__":
    run_router_demo()
