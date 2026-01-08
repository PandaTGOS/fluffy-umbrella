from typing import List, Optional, Any
from dataclasses import dataclass

@dataclass
class Segment:
    start: int
    end: int
    data: Any

class SegmentNode:
    def __init__(self, start, end, segments=None):
        self.start = start
        self.end = end
        self.segments = segments or [] # Segments covering this range
        self.left: Optional[SegmentNode] = None
        self.right: Optional[SegmentNode] = None

class SegmentTree:
    """
    A segment tree to manage document chunks based on character offsets.
    Useful for finding which chunk matches a specific character position or range.
    """
    def __init__(self, segments: List[Segment]):
        # Determine total range
        if not segments:
            self.root = None
            return
            
        min_start = min(s.start for s in segments)
        max_end = max(s.end for s in segments)
        
        self.root = self._build_tree(min_start, max_end, segments)

    def _build_tree(self, start, end, segments):
        if start >= end:
            return None
            
        node = SegmentNode(start, end)
        
        # If this is a leaf node (unit length) or we want to store segments here
        # For simplicity, let's store segments that FULLY cover this node's range
        # But for RAG chunking, we usually want to find "which chunk contains index i".
        # This implementation stores any segment that overlaps with the range.
        
        node.segments = [s for s in segments if max(start, s.start) < min(end, s.end)]
        
        if end - start > 1:
            mid = (start + end) // 2
            node.left = self._build_tree(start, mid, segments)
            node.right = self._build_tree(mid, end, segments)
            
        return node

    def query(self, point: int) -> List[Segment]:
        """
        Find all segments that contain the given point.
        """
        return self._query_recursive(self.root, point)

    def _query_recursive(self, node: SegmentNode, point: int) -> List[Segment]:
        if not node:
            return []
            
        # Segments in this node are candidates, but we should check identifying strict overlap
        # since we pushed overlapping segments down.
        # However, a more standard locking segment tree stores segments at O(log N) nodes.
        # For this simple "chunk finder", a linear scan of pre-filtered segments is fine.
        
        # Actually, let's just descend to the leaf containing the point and collect segments along path?
        # No, the logic above stores ALL overlapping segments at each node which makes the tree huge.
        
        # Better approach for "Point Query":
        # Store segment in the highest node that is fully contained by the segment??
        # Interval Tree is better for this.
        
        res = [s for s in node.segments if s.start <= point < s.end]
        
        # Optimization: Since we stored *overlapping* segments, the node.segments list
        # contains the answer superset. But we need to dedup if we traverse down.
        # Actually, the implementation above stores overlapping at EACH level.
        # So the result at the leaf node (or smallest valid range node) is sufficient?
        # Yes, if we go to the leaf, that leaf's list has everything.
        
        if node.left and point < node.left.end:
            return self._query_recursive(node.left, point)
        elif node.right and point >= node.right.start:
            return self._query_recursive(node.right, point)
            
        return res
