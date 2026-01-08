import sys
import os
import time
from typing import List

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

class ComplianceAuditor:
    def __init__(self):
        self.results = []

    def check(self, name: str, condition: bool, message: str = ""):
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name}: {message}")
        self.results.append((name, status))

def run_audit():
    auditor = ComplianceAuditor()
    print("Starting Enterprise Standards Audit...\n")

    # --- Setup ---
    vector_store = ChromaVectorStore(collection_name="audit_store")
    inverted_index = InvertedIndex()
    rbac = RBACManager()
    router = PrefixRouter()
    graph = ChunkGraph()
    ingestor = ContextualIngestion()
    
    # --- 1. Security & Compliance (RBAC) ---
    print("--- Audit: Security & Access Control ---")
    doc_admin = Document(id="d1", content="Admin Secrets", metadata={"access_required": ["admin"]})
    doc_public = Document(id="d2", content="Public Info", metadata={"access_required": []})
    
    # Mocking filtering logic test
    allowed_admin = rbac.filter_documents(User("u1", [Role.ADMIN]), [doc_admin, doc_public])
    auditor.check("RBAC Admin Access", len(allowed_admin) == 2, "Admin should see all docs")
    
    allowed_guest = rbac.filter_documents(User("u2", [Role.GUEST]), [doc_admin, doc_public])
    auditor.check("RBAC Guest Restriction", len(allowed_guest) == 1 and allowed_guest[0].id == "d2", "Guest should only see public docs")

    # --- 2. Abuse Prevention (Guardrails) ---
    print("\n--- Audit: Abuse Prevention ---")
    router.add_blocked_term("malware")
    is_blocked = router.check_abuse("how to write malware")
    auditor.check("Guardrail Block", is_blocked, "Blocked term 'malware' should be detected")
    
    is_allowed = router.check_abuse("write python code")
    auditor.check("Guardrail Allow", not is_allowed, "Safe query should pass")

    # --- 3. Contextual Integrity (RAG Context) ---
    print("\n--- Audit: Contextual Integrity ---")
    parent = Document(id="p1", content="Parent", metadata={"title": "Q3 Financials"})
    chunk = Document(id="c1", content="Revenue up 5%", metadata={"parent_id": "p1"})
    transformed = ingestor.transform(parent, [chunk])[0]
    
    auditor.check("Context Augmented", "Q3 Financials" in transformed.content, "Chunk content should contain parent title")

    # --- 4. Hybrid Retrieval Completeness ---
    print("\n--- Audit: Hybrid Retrieval ---")
    # Setup standard pipeline
    doc_hybrid = Document(id="h1", content="unique_keyword semantic_concept", metadata={})
    vector_store.add([doc_hybrid])
    inverted_index.add(doc_hybrid)
    
    pipeline = SecureEnterpriseRAG(vector_store, inverted_index, rbac, router, graph)
    
    # Test Keyword Match
    res_kw = pipeline.run("unique_keyword", User("u1", [Role.ADMIN]))
    auditor.check("Keyword Retrieval", len(res_kw.sources) > 0, "Should find doc via keyword")
    
    # Test Semantic Mock (Mock store always returns 0.9 score if query in content in our mock impl)
    # If using real chroma, this depends on embedding.
    # We'll assume the mock behavior we put in verify_rag_demo
    res_sem = pipeline.run("semantic_concept", User("u1", [Role.ADMIN]))
    auditor.check("Semantic Retrieval", len(res_sem.sources) > 0, "Should find doc via semantic search")

    # --- Summary ---
    print("\n--- Audit Summary ---")
    passed = sum(1 for _, s in auditor.results if s == "PASS")
    total = len(auditor.results)
    print(f"Total Standards Checked: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n>> SYSTEM IS COMPLIANT WITH ENTERPRISE STANDARDS <<")
    else:
        print("\n>> SYSTEM COMPLIANCE FAILED <<")

if __name__ == "__main__":
    run_audit()
