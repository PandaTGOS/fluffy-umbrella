from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from brainbox.core.state.rag_state import RAGState
from brainbox.core.context import Context
from brainbox.core.prompts import RAGQAPrompt
from brainbox.core.evaluation.heuristics import evaluate_confidence
from brainbox.core.knowledge.guards import has_answer_evidence
from brainbox.core.config.rag_thresholds import MIN_RETRIEVAL_SUPPORT, MIN_ANSWER_COVERAGE

def retrieve_node(state: RAGState, retriever) -> Dict[str, Any]:
    """Retrieves documents based on the question."""
    result = retriever.retrieve(state.question)
    # Handle both list (simple retrievers) and RetrievalResult (composite)
    if hasattr(result, "documents"):
        docs = result.documents
        signals = getattr(result, "signals", {})
    else:
        docs = result
        signals = {}
        
    return {
        "documents": [
            {
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
                "score": d.score,
            }
            for d in docs
        ],
        "retrieval_signals": signals,
        "retriever_type": retriever.__class__.__name__,
    }

def evidence_guard_node(state: RAGState) -> Dict[str, Any]:
    """Checks if there is evidence in the context to answer the question."""
    if not has_answer_evidence(state.question, state.documents):
        return {"final_decision": "REFUSE_NO_CONTEXT"}
    return {}

def prompt_node(state: RAGState) -> Dict[str, Any]:
    """Builds the prompt specification."""
    prompt = RAGQAPrompt(
        question=state.question,
        context=Context(documents=state.documents),
    ).build()
    return {"prompt_spec": prompt}

def llm_node(state: RAGState, client, tier: str, runtime_options: dict) -> Dict[str, Any]:
    """Executes the LLM call for a specific tier."""
    response = client.generate(
        system_instruction=state.prompt_spec.system_instruction,
        user_input=state.prompt_spec.user_input,
        context=state.prompt_spec.context,
        output_schema=state.prompt_spec.output_schema,
        runtime_options=runtime_options,
    )

    return {
        "response": response,
        "tier": tier,
        "attempts": state.attempts + [{
            "tier": tier,
            "model": response.model_name,
            "runtime_options": runtime_options,
            "confidence": None, # Will be filled by evaluate_node
        }]
    }

def evaluate_node(state: RAGState) -> Dict[str, Any]:
    """Evaluates the confidence of the generated response."""
    confidence = evaluate_confidence(
        state.response.text,
        state.documents,
    )
    
    # Update the last attempt with confidence
    current_attempts = list(state.attempts)
    if current_attempts:
        current_attempts[-1]["confidence"] = confidence
        
    updates = {
        "confidence": confidence,
        "attempts": current_attempts
    }
    
    # Decision Logic
    if (
        confidence.retrieval_support >= MIN_RETRIEVAL_SUPPORT
        and confidence.answer_coverage >= MIN_ANSWER_COVERAGE
    ):
        updates["final_decision"] = f"ACCEPT_{state.tier}"
    elif state.tier == "TIER_2":
        updates["final_decision"] = "REFUSE_LOW_CONFIDENCE"
        
    return updates

def guard_router(state: RAGState):
    if state.final_decision == "REFUSE_NO_CONTEXT":
        return END
    return "prompt"

def eval_router(state: RAGState):
    if state.final_decision:
        return END
    return "llm_tier_2"

def build_rag_graph(retriever, client):
    """Builds and compiles the Adaptive RAG LangGraph."""
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("retrieve", lambda s: retrieve_node(s, retriever))
    graph.add_node("guard", evidence_guard_node)
    graph.add_node("prompt", prompt_node)
    
    # Tier 1 (Fast/Cheap)
    graph.add_node("llm_tier_1", lambda s: llm_node(s, client, "TIER_1", {"temperature": 0.1}))
    graph.add_node("eval_tier_1", evaluate_node)
    
    # Tier 2 (Powerful/Expensive) - e.g. larger model or detailed reasoning
    graph.add_node("llm_tier_2", lambda s: llm_node(s, client, "TIER_2", {"temperature": 0.3}))
    graph.add_node("eval_tier_2", evaluate_node)

    # Edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "guard")
    graph.add_conditional_edges("guard", guard_router)
    
    graph.add_edge("prompt", "llm_tier_1")
    graph.add_edge("llm_tier_1", "eval_tier_1")
    graph.add_conditional_edges("eval_tier_1", eval_router)
    
    graph.add_edge("llm_tier_2", "eval_tier_2")
    graph.add_edge("eval_tier_2", END)

    return graph.compile()
