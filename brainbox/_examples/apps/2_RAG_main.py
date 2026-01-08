from brainbox.core.knowledge import Document, KeywordRetriever
from brainbox.core.llm import OllamaClient
from brainbox.pipelines import RAGPipeline

# Setup Knowledge Base
documents = [
    Document(
        id="france",
        content="France is a country in Europe. Its capital city is Paris.",
        metadata={"source": "wiki"},
    ),
    Document(
        id="germany",
        content="Germany is a European country. Berlin is its capital.",
        metadata={"source": "wiki"},
    ),
    Document(
        id="spain",
        content="Spain is located in Europe and has a rich history.",
        metadata={"source": "wiki"},
    ),
    Document(
        id="italy",
        content="Italy is a European country famous for Rome and ancient history.",
        metadata={"source": "wiki"},
    ),
]

retriever = KeywordRetriever(documents)
client = OllamaClient()
pipeline = RAGPipeline(retriever=retriever, client=client)

questions = [
    "What is the capital of France?",
    "What is the capital of Spain?",
    "What is the capital of Canada?",
]

for q in questions:
    print("=" * 60)
    result = pipeline.run(q)
    print("QUESTION:", result.question)
    print("DECISION:", result.final_decision)
    print("ANSWER:", result.answer)
    print("RETRIEVER:", result.retriever_type)
    print("ATTEMPTS:", len(result.attempts))
    print("TOKENS:", result.token_usage)
    print("=" * 60)
    print()