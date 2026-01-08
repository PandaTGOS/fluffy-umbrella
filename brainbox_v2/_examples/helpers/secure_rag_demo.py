import sys
import os

# Ensure we can import brainbox
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brainbox.core.knowledge.documents import Document
from brainbox.core.vectorstore.chroma import ChromaVectorStore
from brainbox.core.knowledge.indexing.inverted_index import InvertedIndex
from brainbox.core.security.rbac import RBACManager, User, Role
from brainbox.core.routing.prefix_router import PrefixRouter
from brainbox.core.knowledge.graph.chunk_graph import ChunkGraph
from brainbox.core.knowledge.ingestion.contextual import ContextualIngestion
from brainbox.pipelines.secure_enterprise_rag import SecureEnterpriseRAG

def main():
    print("Initializing Secure Enterprise Component...")

    # 1. Setup Components
    # Note: Chroma might create a ./chroma_db folder
    vector_store = ChromaVectorStore(collection_name="demo_secure")
    inverted_index = InvertedIndex()
    rbac = RBACManager()
    router = PrefixRouter()
    graph = ChunkGraph()
    ingestor = ContextualIngestion()

    # 2. Configure Router
    router.add_blocked_term("password")
    router.add_route("salary", "hr_bot")

    # 3. Create Dummy Data
    parent_doc = Document(id="doc1", content="Company Policy", metadata={"title": "HR Policy", "summary": "Rules about leave and conduct."})
    
    chunk1 = Document(id="c1", content="Employees get 20 days leave.", metadata={"parent_id": "doc1", "chunk_index": 0, "access_required": ["member"]})
    chunk2 = Document(id="c2", content="Salary is paid on the 1st.", metadata={"parent_id": "doc1", "chunk_index": 1, "access_required": ["manager"]}) # Higher security
    chunk3 = Document(id="c3", content="Do not share passwords.", metadata={"parent_id": "doc1", "chunk_index": 2, "access_required": ["member"]})

    # 4. Contextual Ingestion & Indexing
    chunks = ingestor.transform(parent_doc, [chunk1, chunk2, chunk3])
    
    print("\n--- Ingestion ---")
    for c in chunks:
        print(f"Indexing Chunk {c.id}: {c.content[:50]}...")
        vector_store.add([c])
        inverted_index.add(c)
    
    graph.build(chunks)

    # 5. Create Pipeline
    pipeline = SecureEnterpriseRAG(vector_store, inverted_index, rbac, router, graph)

    # 6. Run Scenarios
    print("\n--- Scenario 1: Normal Member querying leave ---")
    user_member = User(id="u1", roles=[Role.MEMBER])
    res = pipeline.run("leave days", user_member)
    print(f"Response: {res.response}")
    print(f"Signals: {res.signals}")

    print("\n--- Scenario 2: Member querying blocked term ---")
    res = pipeline.run("show me password", user_member)
    print(f"Response: {res.response}")
    
    print("\n--- Scenario 3: Member querying Manager-only content ---")
    res = pipeline.run("salary details", user_member)
    # The Inverted Index might find "Salary is paid..." but RBAC should filter it out because it requires 'manager' role
    print(f"Response: {res.response}")
    print(f"Docs found but filtered: {res.signals.get('docs_filtered_out', 0)}")

    print("\n--- Scenario 4: Manager querying Manager content ---")
    user_manager = User(id="u2", roles=[Role.MANAGER])
    res = pipeline.run("salary details", user_manager)
    print(f"Response: {res.response}")
    
if __name__ == "__main__":
    main()
