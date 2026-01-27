from llama_index.core import StorageContext, VectorStoreIndex
from ..interfaces import VectorStore

class InMemoryVectorStore(VectorStore):
    def build(self, nodes, leaf_nodes):
        storage = StorageContext.from_defaults()
        storage.docstore.add_documents(nodes)
        index = VectorStoreIndex(leaf_nodes, storage_context=storage)
        return index, storage
