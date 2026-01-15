from typing import Dict, Any, List
from llmkit.interfaces import LLMClient, Pipeline, Context, Retriever, Reranker, Document
from llmkit.guards import has_answer_evidence
from llmkit.prompts import RAGQAPrompt

class EnterpriseRAGPipeline(Pipeline):
    def __init__(
        self, 
        retriever: Retriever, 
        reranker: Reranker,
        client: LLMClient,
        top_k_retrieval: int = 20,
        top_n_rerank: int = 5
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.client = client
        self.top_k_retrieval = top_k_retrieval
        self.top_n_rerank = top_n_rerank

    def run(self, input_data: str) -> Dict[str, Any]:
        question = input_data
        
        print(f"\n[PIPELINE] Processing: {question}")
        
        # 1. Hybrid Retrieval (High Recall)
        print(f"[PIPELINE] Retrieving top {self.top_k_retrieval} candidates...")
        retrieval_result = self.retriever.retrieve(question, k=self.top_k_retrieval)
        documents = retrieval_result.documents
        print(f"[PIPELINE] Retrieved {len(documents)} documents.")
        
        # 2. Listwise Reranking (High Precision)
        # Convert Documents to dicts for reranker (or update Reranker interface to accept Documents?)
        # Base Reranker takes List[Dict]. Let's serialize.
        
        # Helper to serialize
        doc_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score
            }
            for doc in documents
        ]
        
        if self.reranker:
            print(f"[PIPELINE] Reranking to top {self.top_n_rerank}...")
            # Note: Reranker interface was updated to accept query
            reranked_dicts = self.reranker.rerank(doc_dicts, query=question)
            
            # Take top N
            reranked_dicts = reranked_dicts[:self.top_n_rerank]
            print(f"[PIPELINE] Reranked. Top document score/relevance preserved.")
        else:
            reranked_dicts = doc_dicts[:self.top_n_rerank]

        # 3. Build Context
        context = Context(documents=reranked_dicts)
        
        # 4. Guard: Relevance Check
        # Check if we have evidence.
        # We can do this cheaper with a small model or heuristic, 
        # or use the LLM "has_answer_evidence" which calls LLM.
        # For enterprise, we want to fail fast if irrelevant.
        print("[PIPELINE] Verifying relevance...")
        if not has_answer_evidence(question, context.documents):
            print("[PIPELINE] No relevant context found.")
            return {
                "question": question,
                "answer": "I'm sorry, but I couldn't find any information about that in the company policies.",
                "model_name": "N/A",
                "token_usage": {}
            }
            
        # 5. Prompt Engineering with Standard RAG Prompt
        prompt_builder = RAGQAPrompt(question=question, context=context)
        prompt_spec = prompt_builder.build()
        
        # 6. Generation
        print("[PIPELINE] Generating answer...")
        response = self.client.generate(
            system_instruction=prompt_spec.system_instruction,
            user_input=prompt_spec.user_input,
            context=prompt_spec.context,
            output_schema=prompt_spec.output_schema,
            runtime_options={"temperature": 0.1} # Low temp for factual answers
        )
        
        return {
            "question": question,
            "answer": response.text,
            "model_name": response.model_name,
            "token_usage": response.token_usage,
            # Return context metadata for "citations"
            "context_used": [d.get('metadata', {}).get('source', d.get('id', 'Unknown')) for d in reranked_dicts]
        }
