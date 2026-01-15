from typing import List, Dict, Any
from ..interfaces import Reranker, LLMClient
import re

class OllamaListwiseReranker(Reranker):
    def __init__(self, client: LLMClient, top_n: int = 5, model: str = None):
        self.client = client
        self.top_n = top_n
        self.model = model

    def rerank(self, results: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved documents using a listwise approach with an LLM.
        """
        if not results:
            return []
            
        if not query:
            # If no query provided (unlikely in RAG flow if interface passed it), just return top_n
            return results[:self.top_n]

        # Prepare the list of documents for the prompt
        docs_text = ""
        for i, doc in enumerate(results):
            content = doc.get("content", "")[:500] # Truncate for context window safety
            docs_text += f"[{i}] {content}\n\n"

        prompt = f"""
You are an expert Ranker. Your goal is to select the most relevant documents for a given query from a list of candidates.

Query: {query}

Candidate Documents:
{docs_text}

Task:
Rank the top {self.top_n} most relevant documents. 
Output ONLY the indices of the selected documents in order of relevance, like [1, 3, 0].
If a document is not relevant, do not include it.
If fewer than {self.top_n} are relevant, output only those.

Response:
"""
        
        # Call LLM
        # We need to pass system instruction and user input, but here we baked it all into one prompt for simplicity
        # or we can split it. Let's split it slightly.
        
        response = self.client.generate(
            system_instruction="You are a precise ranking assistant. Output only a JSON-style list of indices.",
            user_input=prompt,
            runtime_options={"model": self.model, "temperature": 0.0} if self.model else {"temperature": 0.0}
        )

        output = response.text.strip()
        
        # Parse indices
        try:
            # Look for something like [0, 1, 2]
            match = re.search(r"\[([\d,\s]+)\]", output)
            if match:
                indices_str = match.group(1)
                indices = [int(idx.strip()) for idx in indices_str.split(",") if idx.strip().isdigit()]
            else:
                # Fallback: look for just digits
                indices = []
                # Maybe they just wrote "1, 2, 3"
                parts = output.split(',')
                for p in parts:
                    clean = p.strip()
                    if clean.isdigit():
                        indices.append(int(clean))
        except Exception:
            print(f"[RERANKER] Failed to parse LLM output: {output}")
            return results[:self.top_n]

        # Construct reranked list
        reranked_results = []
        seen_indices = set()
        
        for idx in indices:
            if 0 <= idx < len(results) and idx not in seen_indices:
                # Assign a score based on rank? Or keep original?
                # Let's just reorder.
                reranked_results.append(results[idx])
                seen_indices.add(idx)
        
        # If we need more to fill top_n (or if LLM returned nothing relevant but we want to return something)
        # For enterprise RAG, usually we want strict relevance, but fallback is safe.
        # Let's append the rest of original top results if we haven't met top_n yet, 
        # BUT only if we want high recall fallback. The prompt said "If not relevant, do not include".
        # So we might trust the LLM. 
        # However, if LLM fails completely (empty list), we should fallback to original order.
        
        if not reranked_results:
             return results[:self.top_n]

        return reranked_results
