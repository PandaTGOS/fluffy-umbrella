from typing import Dict, List, Any, Tuple
import collections

class SimpleGraphStore:
    """
    A simple in-memory graph store.
    """
    def __init__(self):
        # Adjacency list: node_id -> list of (target_node_id, relationship_type)
        self.adj_list: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
        self.nodes: Dict[str, Any] = {} # Store node data

    def add_node(self, node_id: str, data: Any = None):
        self.nodes[node_id] = data

    def add_edge(self, source_id: str, target_id: str, relation: str):
        if source_id not in self.nodes:
            self.add_node(source_id)
        if target_id not in self.nodes:
            self.add_node(target_id)
            
        self.adj_list[source_id].append((target_id, relation))

    def get_neighbors(self, node_id: str, relation: str = None) -> List[Tuple[str, Any]]:
        if node_id not in self.adj_list:
            return []
            
        neighbors = []
        for target, rel in self.adj_list[node_id]:
            if relation is None or rel == relation:
                neighbors.append((target, self.nodes[target]))
        return neighbors

    def get_path(self, start_node: str, end_node: str, max_depth: int = 3) -> List[str]:
        # Simple BFS
        queue = [(start_node, [start_node])]
        visited = set()
        
        while queue:
            (vertex, path) = queue.pop(0)
            if vertex in visited:
                continue
            
            visited.add(vertex)
            
            for neighbor, _ in self.adj_list[vertex]:
                if neighbor == end_node:
                    return path + [end_node]
                
                if len(path) < max_depth:
                    queue.append((neighbor, path + [neighbor]))
                    
        return []
