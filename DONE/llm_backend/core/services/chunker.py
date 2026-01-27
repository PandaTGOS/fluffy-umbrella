from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes
)

def build_chunks(documents, chunk_sizes):

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes
    )

    nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)

    return nodes, leaf_nodes