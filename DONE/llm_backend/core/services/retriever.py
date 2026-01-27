from typing import Optional

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.prompts import PromptTemplate
from langdetect import detect


DEFAULT_TOP_K = 10

DEFAULT_QA_TEMPLATE = """INSTRUCTIONS:
Provide a detailed, structured answer:
- Comprehensive
- Concise
- Include links and forms if necessary

CRITICAL RULES:
- NEVER invent information.
- Use ONLY the provided context.
- For irrelevant queries that are not at all related to HR reply with:
  "This query is out of my scope, kindly contact the concerned HR person"

Now provide the answer:
Question: {query}

Context:
{context}
"""


def detect_language(text: str):
    try:
        lang = detect(text)
        return "sv" if lang.startswith("sv") else "en"
    except Exception:
        return "en"
    

def _normalize_template(template: Optional[str]) -> str:
    if not template:
        return DEFAULT_QA_TEMPLATE

    # Map custom placeholders TO standard ones
    template = template.replace("{context}", "{context_str}")
    template = template.replace("{query}", "{query_str}")
    template = template.replace("{question}", "{query_str}")

    if "{context_str}" not in template or "{query_str}" not in template:
        return DEFAULT_QA_TEMPLATE

    return template


def build_query_engine(
    query: str,
    index,
    storage,
    *,
    top_k: Optional[int] = None,
    qa_template: Optional[str] = None,
    use_hybrid: bool = False,
    reranker_config = None
):
    lang = detect_language(query)

    filters = MetadataFilters(filters=[
        MetadataFilter(key="language", value=lang)
    ])

    # Hybrid Search Configuration
    vector_store_kwargs = {}
    if use_hybrid:
        vector_store_kwargs["vector_store_query_mode"] = "hybrid"

    base_retriever = index.as_retriever(
        similarity_top_k=top_k or DEFAULT_TOP_K,
        filters=filters,
        **vector_store_kwargs
    )

    merging_retriever = AutoMergingRetriever(
        base_retriever,
        storage,
        verbose=False
    )

    # Reranking Configuration
    node_postprocessors = []
    if reranker_config and reranker_config.provider != "none":
        # Dynamic import to avoid hard dependency if not used
        if reranker_config.provider == "colbert":
            try:
                from llama_index.postprocessor.colbert_rerank import ColbertRerank
                reranker = ColbertRerank(
                    top_n=reranker_config.top_n,
                    model=reranker_config.model or "colbert-ir/colbertv2.0"
                )
                node_postprocessors.append(reranker)
            except ImportError:
                print("Warning: ColbertRerank not installed.")
        # Add Cohere block here similarly if needed

    synthesizer = CompactAndRefine(
        text_qa_template=PromptTemplate(_normalize_template(qa_template))
    )

    return RetrieverQueryEngine(
        retriever=merging_retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=node_postprocessors
    )

