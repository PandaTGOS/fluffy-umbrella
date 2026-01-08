from brainbox.core.knowledge import Document, KeywordRetriever
from brainbox.pipelines import RAGPipeline
from brainbox.core.llm import OllamaClient

def run_high_confidence_test():
    # 1. Setup Data for High Overlap
    # Using a "password" scenario where the answer is likely to be a direct extraction of the text.
    question = "What is the capital of SecretLand? Answer with ONLY the name. Do not provide reasoning."

    print(f"Running Pipeline for: '{question}'")
    print("-" * 60)
    
    documents = [
        Document(
            id="secret_doc",
            content="The capital of SecretLand is Blueberry.",
            metadata={"source": "geography_manual"},
        ),
        Document(
            id="decoys",
            content="Bananas are yellow. Apples are red.",
            metadata={"source": "fruit_facts"},
        ),
    ]

    retriever = KeywordRetriever(documents)
    # Use Real Ollama Client
    client = OllamaClient()
    pipeline = RAGPipeline(retriever=retriever, client=client)
    
    result = pipeline.run(question)

    # 3. Inspect Results
    print(f"DECISION: {result.final_decision}")
    print(f"ANSWER: {result.answer}")
    print(f"RETRIEVER: {result.retriever_type}")
    print("-" * 60)
    
    # 4. Deep Dive into Attempts
    if result.attempts:
        print("ATTEMPT DETAILS:")
        for i, attempt in enumerate(result.attempts, 1):
            print(f"\nAttempt {i} (Tier {attempt['tier']}):")
            print(f"  Model: {attempt.get('model')}")
            conf = attempt.get('confidence')
            if conf:
                print(f"  Retrieval Support: {conf.retrieval_support:.2f}")
                print(f"  Answer Coverage:   {conf.answer_coverage:.2f}")
                if conf.metadata:
                    print(f"  Metadata: {conf.metadata}")
            else:
                print("  Confidence: None")

    print("=" * 60)

if __name__ == "__main__":
    run_high_confidence_test()
