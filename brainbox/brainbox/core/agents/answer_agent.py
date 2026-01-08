from typing import Dict, Any, TYPE_CHECKING
from brainbox.core.agents.base import Agent
from brainbox.core.llm import LLMResponse

if TYPE_CHECKING:
    from brainbox.core.llm.llm_client import LLMClient

class AnswerAgent(Agent):
    name = "answer_agent"

    def __init__(self, client: "LLMClient"):
        self.client = client

    def run(self, state) -> Dict[str, Any]:
        """
        Generates a direct answer or formats a final response.
        Used when no tools or RAG are explicitly needed (or as fallback).
        """
        # Simple Prompt
        system_inst = "You are a helpful assistant. Answer the user's question directly and concisely."
        response = self.client.generate(
            system_instruction=system_inst,
            user_input=state.question,
            runtime_options={"temperature": 0.5}
        )
        
        return {
            "response": response,
            "final_decision": "ANSWER_READY"
        }
