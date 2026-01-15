from typing import Dict, Any
from llmkit.interfaces import LLMClient, Pipeline, Context, Retriever
from llmkit.guards import has_answer_evidence
from llmkit.prompts import RAGQAPrompt

class RAGPipeline(Pipeline):
    def __init__(self, retriever: Retriever, client: LLMClient):
        self.retriever = retriever
        self.client = client

    def run(self, input_data: str) -> Dict[str, Any]:
        question = input_data
        
        # 1. Retrieve Knowledge
        retrieval_result = self.retriever.retrieve(question)
        documents = retrieval_result.documents
        
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
                "model_name": "N/A (LLM not called)",
                "token_usage": {"total": 0},
            }
            
        # 4. Prompt Engineering
        prompt_builder = RAGQAPrompt(question=question, context=context)
        prompt_spec = prompt_builder.build()
        
        # 5. Execution (Call LLM)
        response = self.client.generate(
            system_instruction=prompt_spec.system_instruction,
            user_input=prompt_spec.user_input,
            context=prompt_spec.context,
            output_schema=prompt_spec.output_schema,
        )
        
        return {
            "question": question,
            "answer": response.text,
            "model_name": response.model_name,
            "token_usage": response.token_usage,
        }