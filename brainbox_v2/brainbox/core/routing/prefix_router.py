from typing import Dict, Any, Optional

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.handler: Optional[Any] = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, handler: Any = None):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.handler = handler

    def search(self, word: str) -> Optional[Any]:
        """
        Exact match search. Returns handler if found.
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end_of_word:
            return node.handler
        return None

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

class PrefixRouter:
    """
    Routes queries based on prefix matching or blocks specific terms (abuse prevention).
    """
    def __init__(self):
        self.routes = Trie()
        self.abuse_list = Trie()

    def add_route(self, prefix: str, handler: str):
        """
        Add a routing rule. E.g. "finance" -> "finance_bot"
        """
        self.routes.insert(prefix, handler)

    def add_blocked_term(self, term: str):
        """
        Add a term to the blocklist.
        """
        self.abuse_list.insert(term, True)

    def check_abuse(self, text: str) -> bool:
        """
        Check if text contains any blocked terms.
        Naive implementation: checks if any word in text starts with a blocked term.
        """
        words = text.split()
        for word in words:
            # Check if this word matches any blocked term in Trie
            # This logic assumes "kill" blocks "kill", but maybe not "skill" depending on implementation.
            # My Trie implementation supports full word lookup or prefix.
            # Let's check if the word IS in the abuse list.
            if self.abuse_list.search(word):
                return True
        return False
        
    def route(self, query: str) -> Optional[str]:
        """
        Predict route based on query prefix.
        """
        # A simple logic: Check if the first word matches a route
        first_word = query.split()[0] if query else ""
        return self.routes.search(first_word)
