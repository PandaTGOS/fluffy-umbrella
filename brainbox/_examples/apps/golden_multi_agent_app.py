from brainbox.core.tools.registry import ToolRegistry
from brainbox.core.tools.calculator import CalculatorTool
from brainbox.core.llm import OllamaClient
from apps.golden_rag_app import main as run_rag_demo
# Reuse components or redefine for clarity? Redefine for standalone.
from brainbox.core.knowledge import Document
from brainbox.core.knowledge.retrievers import KeywordRetriever
from brainbox.pipelines.agent_pipeline import AgentPipeline

def main():
    print("🌟 Running Golden Multi-Agent App...")
    
    # -----------------------------
    # 1. Setup Tools (The "Doer" Capability)
    # -----------------------------
    # Register tools globally (or scoped if we improved registry)
    calculator_tool = CalculatorTool()
    ToolRegistry.register(calculator_tool)
    
    # -----------------------------
    # 2. Setup Resources
    # -----------------------------
    llm = OllamaClient()
    
    # Minimal RAG for fallback/hybrid
    docs = [Document(id="1", content="Sakhi works at Google.", metadata={})]
    retriever = KeywordRetriever(docs) # Simple for demo
    
    # -----------------------------
    # 3. Pipeline
    # -----------------------------
    # The pipeline inherently supports routing to Tools, RAG, or AnswerAgent
    pipeline = AgentPipeline(retriever=retriever, client=llm)
    
    # -----------------------------
    # 4. Scenarios
    # -----------------------------
    
    # Scenario A: Tool (Calc)
    print("\n--- Scenario A: Tool Agent ---")
    res_tool = pipeline.run("Calculate 55 * 4")
    print("A Answer:", res_tool.answer)
    print("A Decision:", res_tool.final_decision)
    
    # Scenario B: General Chat (Answer Agent)
    print("\n--- Scenario B: Answer Agent ---")
    res_chat = pipeline.run("Tell me a funny joke about coding.")
    print("B Answer:", res_chat.answer)
    
    # Scenario C: RAG (Retrieval Agent)
    print("\n--- Scenario C: Retrieval Agent ---")
    res_rag = pipeline.run("Where does Sakhi work?")
    print("C Answer:", res_rag.answer)

if __name__ == "__main__":
    main()
