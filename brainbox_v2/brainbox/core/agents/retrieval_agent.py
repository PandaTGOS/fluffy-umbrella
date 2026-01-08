from typing import Dict, Any, TYPE_CHECKING
from brainbox.core.agents.base import Agent
from brainbox.core.prompts import RAGQAPrompt
from brainbox.core.context import Context

if TYPE_CHECKING:
    from brainbox.core.llm.llm_client import LLMClient

class RetrievalAgent(Agent):
    name = "retrieval_agent"

    def __init__(self, client: "LLMClient"):
        self.client = client

    def run(self, state) -> Dict[str, Any]:
        """
        Retrieval Agent Logic:
        1. Build Prompt using Context (documents).
        2. Generate Answer.
        3. Attempts tracking logic (optional, preserved from llm_node).
        """
        prompt = RAGQAPrompt(
            question=state.question,
            context=Context(documents=state.documents)
        ).build()

        response = self.client.generate(
            system_instruction=prompt.system_instruction,
            user_input=prompt.user_input,
            context=prompt.context,
            runtime_options={"temperature": 0.1} # Low temp for RAG
        )

        return {
            "response": response,
            "final_decision": "ANSWER_READY"
        }
