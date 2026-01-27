from ...services.retriever import build_query_engine, detect_language
from ...services.llm_guardrails import LLMScopeGuard

_llm_scope_guard = None

def get_scope_guard():
    global _llm_scope_guard

    if _llm_scope_guard is None:
        _llm_scope_guard = LLMScopeGuard()

    return _llm_scope_guard


def is_in_scope(nodes):
    if not nodes:
        return False
    return len(" ".join(nodes).strip()) >= 60


async def node_detect_language(state):
    return {"language": detect_language(state["query"])}


async def node_retrieve(state, retriever):

    retrieved = await retriever.aretrieve(state["query"])

    contents, sources = [], set()

    for n in retrieved:
        contents.append(n.get_content())

        if "filename" in n.metadata:
            sources.add(n.metadata["filename"])

    return {
        "retrieved_nodes": contents,
        "sources": list(sources)
    }



async def node_synthesize(state, index, storage, cfg):
    if not is_in_scope(state["retrieved_nodes"]):
        return {"answer": ("This query is out of the HR scope, kindly contact the concerned HR person.")}

    engine = build_query_engine(
        state["query"],
        index,
        storage,
        top_k=cfg.rag.top_k,
        qa_template=cfg.rag.qa_template,
        use_hybrid=cfg.rag.use_hybrid_search,
        reranker_config=cfg.rag.reranker
    )

    answer = await engine.aquery(state["query"])
    return {"answer": str(answer)}


async def node_scope_guard(state):
    query = state["query"]
    guard = get_scope_guard()

    if not await guard.is_relevant(query):
        return {
            "blocked": True,
            "answer": (
                "This query is out of scope. You will need to contact Function.HRHelp@gknaerospace.com for this matter."
            ),
        }

    return {"blocked": False}