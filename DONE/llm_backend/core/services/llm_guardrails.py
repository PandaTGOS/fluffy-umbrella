from llama_index.core import Settings


SCOPE_PROMPT = """
You are a classifier for an HR assistant. 

Decide if the below user query is ethical and related to HR queries only: 
- HR policies 
- salary 
- benefits 
- leave 
- employment 
- payroll 
- workplace rules 
- employee relations
- medical proofs 

Additional decision rule:
- If the query refers to previous context - the decision is YES.
Example "Give me the link for that" or "How does it work" etc.
 
Reply with ONLY one word:

YES or NO 

User Query: 
{query}
"""


class LLMScopeGuard:
    def __init__(self, model=None):
        self.llm = model or Settings.llm
        self._llm = None

        if self.llm is None:
            raise RuntimeError("LLM not configured")

    async def is_relevant(self, query: str) -> bool:
        prompt = SCOPE_PROMPT.format(query=query)

        response = await self.llm.acomplete(prompt)

        answer = str(response).strip().upper()

        return answer.startswith("YES")
