# RAG Context Fix Instructions

## Issue
The RAG system was asking for context because the `retriever.py` service was using non-standard placeholders (`{context}`, `{query}`) in the prompt template, while the LlamaIndex engine expects `{context_str}` and `{query_str}`. This caused the engine to fail injecting the retrieved chunks.

## Fix
Apply the following changes to `llm_backend/core/services/retriever.py`:

### 1. Update Default Template
Change the placeholders in `DEFAULT_QA_TEMPLATE` to use `_str` suffixes.

```python
# OLD
DEFAULT_QA_TEMPLATE = """...
Question: {query}

Context:
{context}
"""

# NEW
DEFAULT_QA_TEMPLATE = """...
Question: {query_str}

Context:
{context_str}
"""
```

### 2. Update Template Normalization
Modify `_normalize_template` to ensure we map *to* the standard placeholders, not away from them.

**Old Logic (Incorrect):**
```python
def _normalize_template(template: Optional[str]) -> str:
    if not template:
        return DEFAULT_QA_TEMPLATE

    # This BROKE standard templates by renaming valid placeholders to invalid ones
    template = template.replace("{context_str}", "{context}")
    template = template.replace("{query_str}", "{query}")
    template = template.replace("{question}", "{query}")

    if "{context}" not in template or "{query}" not in template:
        return DEFAULT_QA_TEMPLATE

    return template
```

**New Logic (Correct):**
```python
def _normalize_template(template: Optional[str]) -> str:
    if not template:
        return DEFAULT_QA_TEMPLATE

    # Map custom placeholders TO standard ones
    template = template.replace("{context}", "{context_str}")
    template = template.replace("{query}", "{query_str}")
    template = template.replace("{question}", "{query_str}")

    # Check for validity using standard keys
    if "{context_str}" not in template or "{query_str}" not in template:
        return DEFAULT_QA_TEMPLATE

    return template
```
