from typing import List, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass

class Role(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    MEMBER = "member"
    GUEST = "guest"

@dataclass
class User:
    id: str
    roles: List[Role]
    metadata: Dict[str, str] = None

class RBACManager:
    """
    Manages Role-Based Access Control logic.
    """
    def __init__(self):
        # Define permissions for each role
        self.permissions: Dict[Role, Set[str]] = {
            Role.ADMIN: {"read", "write", "delete", "manage_users", "manage_system"},
            Role.MANAGER: {"read", "write", "manage_users"},
            Role.MEMBER: {"read", "write"},
            Role.GUEST: {"read"},
        }

    def can(self, user: User, action: str) -> bool:
        """
        Check if a user has permission to perform an action.
        """
        for role in user.roles:
            if role in self.permissions and action in self.permissions[role]:
                return True
        return False

    def filter_documents(self, user: User, documents: List[Dict]) -> List[Dict]:
        """
        Filter a list of documents based on user access.
        This assumes documents have an 'access_control' field in metadata.
        If access_control is missing, assume public (or default deny depending on policy).
        Here we assume public if missing for simplicity, or explicit role required.
        """
        allowed_docs = []
        for doc in documents:
            # Handle both Document objects and dictionaries
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
            else:
                metadata = doc.metadata if doc.metadata else {}
            
            required_roles = metadata.get("access_required", [])
            
            if not required_roles:
                # No specific restriction
                allowed_docs.append(doc)
                continue

            # Check if user has ANY of the required roles
            # Note: converting string roles in metadata to Role enums if needed
            user_role_values = {r.value for r in user.roles}
            if any(req in user_role_values for req in required_roles):
                allowed_docs.append(doc)
        
        return allowed_docs

# Singleton instance or factory can be used
rbac_manager = RBACManager()
