from typing import Dict, Any
from brainbox.core.llm import LLMClient
from brainbox.core.knowledge import BaseRetriever
from brainbox.core.context import Context
from brainbox.core.prompts import RAGQAPrompt
from brainbox.core.knowledge.guards import has_answer_evidence
from brainbox.core.evaluation.heuristics import evaluate_confidence
from .base import Pipeline


MIN_RETRIEVAL_SUPPORT = 0.3
MIN_ANSWER_COVERAGE = 0.5

TIER_1_OPTIONS = {"temperature": 0.1}
TIER_2_OPTIONS = {"temperature": 0.3}


class RAGPipeline(Pipeline):
    def __init__(self, retriever: BaseRetriever, client: LLMClient):
        self.retriever = retriever
        self.client = client

    def _call_llm(self, prompt_spec, runtime_options=None):
        return self.client.generate(
            system_instruction=prompt_spec.system_instruction,
            user_input=prompt_spec.user_input,
            context=prompt_spec.context,
            output_schema=prompt_spec.output_schema,
            runtime_options=runtime_options,
        )

    def run(self, input_data: str) -> Dict[str, Any]:
        question = input_data
        
        # 1. Retrieve Knowledge
        documents = self.retriever.retrieve(question)
        
        # 2. Build Context
        structured_docs = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score,
            }
            for doc in documents
        ]
        context = Context(documents=structured_docs)
        
        # 3. Guard
        if not has_answer_evidence(question, context.documents):
            return {
                "question": question,
                "answer": "I do not know (insufficient context)",
                "confidence": None,
                "model_name": "N/A (LLM not called)",
                "token_usage": {"total": 0},
                "attempts": [],
            }
            
        # 4. Prompt Engineering
        prompt_builder = RAGQAPrompt(question=question, context=context)
        prompt_spec = prompt_builder.build()
        
        attempts = []
        
        # 5 & 6. Execution & Evaluation (Tiered)
        
        # Tier 1
        response = self._call_llm(prompt_spec, runtime_options=TIER_1_OPTIONS)
        confidence = evaluate_confidence(response.text, context.documents)
        
        attempts.append({
            "tier": 1,
            "confidence": confidence,
            "runtime_options": TIER_1_OPTIONS
        })
        
        if (
            confidence.retrieval_support >= MIN_RETRIEVAL_SUPPORT
            and confidence.answer_coverage >= MIN_ANSWER_COVERAGE
        ):
             return {
                "question": question,
                "answer": response.text,
                "confidence": confidence,
                "model_name": response.model_name,
                "token_usage": response.token_usage,
                "attempts": attempts,
            }

        # Tier 2 (Retry with different options if Tier 1 failed confidence check)
        response = self._call_llm(prompt_spec, runtime_options=TIER_2_OPTIONS)
        confidence = evaluate_confidence(response.text, context.documents)
        
        attempts.append({
            "tier": 2,
            "confidence": confidence,
            "runtime_options": TIER_2_OPTIONS
        })
        
        if (
            confidence.retrieval_support >= MIN_RETRIEVAL_SUPPORT
            and confidence.answer_coverage >= MIN_ANSWER_COVERAGE
        ):
             return {
                "question": question,
                "answer": response.text,
                "confidence": confidence,
                "model_name": response.model_name,
                "token_usage": response.token_usage,
                "attempts": attempts,
            }
        
        # Fail
        return {
            "question": question,
            "answer": "I do not know (low confidence)",
            "confidence": confidence,
            "model_name": response.model_name,
            "token_usage": response.token_usage,
            "attempts": attempts,
        }
