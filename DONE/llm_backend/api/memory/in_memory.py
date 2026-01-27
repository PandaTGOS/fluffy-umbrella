from typing import Dict
from collections import defaultdict
from .interface import SessionMemory

SUMMARY_PROMPT = """
You are maintaining a rolling summary of a conversation.

Current summary:
{summary}

New interaction:
User: {user}
Assistant: {assistant}

Update the summary so that it captures ONLY the important context
needed to answer future questions.

Rules:
- Be concise
- Do not repeat wording
- Do not invent facts
- Focus on user intent and constraints

Updated summary:
"""

class InMemoryRollingSummary(SessionMemory):
    def __init__(self):
        self._summaries: Dict[str, str] = defaultdict(str)

    def get_summary(self, session_id: str) -> str:
        return self._summaries.get(session_id, "")

    def update(self, session_id, user_query, assistant_answer, llm):
        prompt = SUMMARY_PROMPT.format(
            summary=self._summaries.get(session_id, ""),
            user=user_query,
            assistant=assistant_answer
        )

        updated_summary = llm.complete(prompt).text
        self._summaries[session_id] = updated_summary.strip()