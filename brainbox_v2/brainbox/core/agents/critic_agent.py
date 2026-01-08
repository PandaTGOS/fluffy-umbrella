from typing import Dict, Any
from brainbox.core.agents.base import Agent
from brainbox.core.evaluation.heuristics import evaluate_confidence

class CriticAgent(Agent):
    name = "critic_agent"

    def run(self, state) -> Dict[str, Any]:
        """
        Critic Agent Logic:
        Evaluates the quality of the current response against retrieved documents.
        """
        if not state.response:
             # Cannot critique if no response
             return {}
             
        confidence = evaluate_confidence(
            state.response.text,
            state.documents
        )

        return {"confidence": confidence}
