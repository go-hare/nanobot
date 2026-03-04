from __future__ import annotations

from pathlib import Path

from nanobot.memory.affective_store import AffectiveStore
from nanobot.memory.policy_state_store import PolicyStateStore
from nanobot.memory.relational_store import RelationalStore
from nanobot.memory.semantic_store import SemanticStore


class MemoryFacade:
    """Unified access to semantic/relational/affective stores."""

    def __init__(self, workspace: Path):
        self.semantic = SemanticStore(workspace)
        self.relational = RelationalStore(workspace)
        self.affective = AffectiveStore(workspace)
        self.policy_state = PolicyStateStore(workspace)

    def save_policy_adjustment(self, adjustment: dict) -> None:
        self.policy_state.save_adjustment(adjustment)

    def load_policy_adjustment(self) -> dict | None:
        return self.policy_state.load_active_adjustment()

