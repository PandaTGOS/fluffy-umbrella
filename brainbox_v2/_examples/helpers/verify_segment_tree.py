import sys
import os

# Ensure we can import brainbox
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from brainbox.core.knowledge.indexing.segment_tree import SegmentTree, Segment

def test_segment_tree():
    print("Testing Segment Tree...")
    
    # Create segments (e.g. chunks mapping to character ranges in a document)
    # 0-10: Chunk A
    # 10-20: Chunk B
    # 5-15: Chunk C (Overlapping)
    
    segments = [
        Segment(0, 10, "Chunk A"),
        Segment(10, 20, "Chunk B"),
        Segment(5, 15, "Chunk C") 
    ]
    
    tree = SegmentTree(segments)
    
    # Test Case 1: Point 2 (Only A)
    res = tree.query(2)
    print(f"Query(2): {[s.data for s in res]}")
    assert "Chunk A" in [s.data for s in res]
    assert "Chunk B" not in [s.data for s in res]
    
    # Test Case 2: Point 12 (B and C)
    res = tree.query(12)
    print(f"Query(12): {[s.data for s in res]}")
    assert "Chunk B" in [s.data for s in res]
    assert "Chunk C" in [s.data for s in res]
    assert "Chunk A" not in [s.data for s in res]

    # Test Case 3: Point 7 (A and C)
    res = tree.query(7)
    print(f"Query(7): {[s.data for s in res]}")
    assert "Chunk A" in [s.data for s in res]
    assert "Chunk C" in [s.data for s in res]
    
    print("Segment Tree Verification Passed!")

if __name__ == "__main__":
    test_segment_tree()
