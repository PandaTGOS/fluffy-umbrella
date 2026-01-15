from llmkit.llms import OllamaClient
from llmkit.embeddings import OllamaEmbeddingClient
from llmkit.vectorstores import InMemoryVectorStore
from llmkit.chunkers import RecursiveChunker
from llmkit.knowledgebases import DirectoryKnowledgeBase
from llmkit.rerankers import OllamaListwiseReranker
from pipelines.enterprise_pipeline import EnterpriseRAGPipeline

from dotenv import load_dotenv
load_dotenv()


print("Running Enterprise RAG Ingestion")

# 1. Improved Chunking (Hierarchy Aware)
chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=64, keep_headers=True)
vector_store = InMemoryVectorStore()

embeddings = OllamaEmbeddingClient()
llm = OllamaClient(default_model="qwen:1.8b") # Ensure model is set

print("\nScanning and Indexing...")
# 2. Hybrid Ingestion (Vector + BM25)
kb = DirectoryKnowledgeBase.from_path(
    path="./data/HR_Bot_Data/english_data",
    chunker=chunker,
    embedding_client=embeddings,
    vector_store=vector_store,
    enable_bm25=True
)

# 3. Reranker
reranker = OllamaListwiseReranker(client=llm, top_n=5, model="qwen:1.8b")

# 4. Enterprise Pipeline
pipeline = EnterpriseRAGPipeline(
    retriever=kb.as_retriever(),
    reranker=reranker,
    client=llm,
    top_k_retrieval=20, # Get more for reranking
    top_n_rerank=5
)

questions = [
    "Who do i contact for industry relations advice",
    "How is the vacation factor calculated if I work 4 days a week?",
    "What is the capital of Poland?"
]

for q in questions:
    print("-" * 60)
    result = pipeline.run(q)
    print(f"\nFinal Answer:\n{result['answer']}")
    
    if result.get("context_used"):
         print(f"\nSources: {result['context_used']}")
         
    print("-" * 60 + "\n")