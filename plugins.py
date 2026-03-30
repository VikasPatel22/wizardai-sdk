"""
WizardAI Memory Manager
-----------------------
Provides short-term and long-term memory for conversational agents.
Supports in-memory storage and optional JSON/pickle persistence.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from .utils import DataSerializer, Logger


class Message:
    """Represents a single conversation message.

    Attributes:
        role:      Speaker role – 'user', 'assistant', or 'system'.
        content:   The message text or payload.
        timestamp: Unix timestamp when the message was created.
        metadata:  Arbitrary key-value pairs attached to the message.
    """

    __slots__ = ("role", "content", "timestamp", "metadata")

    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        msg = cls(data["role"], data["content"], data.get("metadata", {}))
        msg.timestamp = data.get("timestamp", time.time())
        return msg

    def __repr__(self):
        preview = self.content[:40] + "…" if len(self.content) > 40 else self.content
        return f"Message(role={self.role!r}, content={preview!r})"


class MemoryManager:
    """Manages short-term conversation history and long-term key-value memory.

    Short-term memory holds the sliding window of recent messages that is
    passed to the AI on each turn.  Long-term memory stores arbitrary
    facts as a persistent key-value store.

    Example::

        mem = MemoryManager(max_history=20)
        mem.add_message("user", "What's the weather like?")
        mem.add_message("assistant", "It's sunny today!")
        mem.remember("user_name", "Alice")

        history = mem.get_history()         # last 20 messages
        name    = mem.recall("user_name")   # "Alice"
        mem.save("session.json")
        mem.load("session.json")
    """

    def __init__(
        self,
        max_history: int = 50,
        persist_path: Optional[Union[str, Path]] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            max_history:  Maximum number of messages kept in short-term memory.
            persist_path: If provided, memory is auto-saved to this path on
                          every write operation.
            logger:       Optional Logger instance.
        """
        self.max_history = max_history
        self.persist_path = Path(persist_path) if persist_path else None
        self.logger = logger or Logger("MemoryManager")
        self._serializer = DataSerializer()

        self._history: Deque[Message] = deque(maxlen=max_history)
        self._long_term: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}  # ephemeral session-level context

        if self.persist_path and self.persist_path.exists():
            self.load(self.persist_path)

    # ------------------------------------------------------------------
    # Short-term (conversation history)
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Append a message to the short-term history.

        Args:
            role:     'user' | 'assistant' | 'system'.
            content:  Message text.
            metadata: Optional extra data attached to the message.

        Returns:
            The created :class:`Message` object.
        """
        msg = Message(role, content, metadata)
        self._history.append(msg)
        self.logger.debug(f"[Memory] +{role}: {content[:60]!r}")
        self._auto_save()
        return msg

    def get_history(
        self,
        n: Optional[int] = None,
        role_filter: Optional[str] = None,
    ) -> List[Message]:
        """Return recent conversation history.

        Args:
            n:           Return only the last *n* messages (default: all).
            role_filter: If set, only return messages with this role.

        Returns:
            List of :class:`Message` objects (oldest first).
        """
        msgs = list(self._history)
        if role_filter:
            msgs = [m for m in msgs if m.role == role_filter]
        if n is not None:
            msgs = msgs[-n:]
        return msgs

    def get_history_as_dicts(self, n: Optional[int] = None) -> List[Dict]:
        """Return history as a list of dicts (suitable for API payloads)."""
        return [m.to_dict() for m in self.get_history(n)]

    def get_messages_for_api(
        self,
        n: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Return history formatted for OpenAI / Anthropic message arrays.

        Returns a list like::

            [{"role": "user", "content": "Hello"}, ...]

        Args:
            n:              Limit to last *n* messages.
            include_system: Include system messages in the output.
        """
        msgs = self.get_history(n)
        if not include_system:
            msgs = [m for m in msgs if m.role != "system"]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def clear_history(self):
        """Wipe short-term conversation history."""
        self._history.clear()
        self.logger.debug("[Memory] Short-term history cleared.")
        self._auto_save()

    def last_message(self, role: Optional[str] = None) -> Optional[Message]:
        """Return the most recent message, optionally filtered by role."""
        msgs = list(self._history)
        if role:
            msgs = [m for m in msgs if m.role == role]
        return msgs[-1] if msgs else None

    def search_history(self, query: str, top_k: int = 5) -> List[Tuple[Message, float]]:
        """Simple keyword-based search over conversation history.

        Returns up to *top_k* (message, relevance_score) tuples, sorted by
        descending relevance.  Relevance is a basic TF-like overlap score.

        Args:
            query: Search query string.
            top_k: Maximum number of results.
        """
        query_words = set(query.lower().split())
        results: List[Tuple[Message, float]] = []

        for msg in self._history:
            words = set(msg.content.lower().split())
            overlap = len(query_words & words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                results.append((msg, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any):
        """Store a fact in long-term memory.

        Args:
            key:   Unique identifier for the fact.
            value: Any JSON-serialisable Python object.
        """
        self._long_term[key] = value
        self.logger.debug(f"[Memory] Stored long-term: {key!r}")
        self._auto_save()

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact from long-term memory.

        Args:
            key:     The fact identifier.
            default: Value to return if the key is not found.
        """
        return self._long_term.get(key, default)

    def forget(self, key: str) -> bool:
        """Remove a fact from long-term memory.

        Returns:
            True if the key existed and was removed, False otherwise.
        """
        if key in self._long_term:
            del self._long_term[key]
            self._auto_save()
            return True
        return False

    def list_memories(self) -> List[str]:
        """Return a list of all long-term memory keys."""
        return list(self._long_term.keys())

    # ------------------------------------------------------------------
    # Ephemeral context (current session only)
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: Any):
        """Set an ephemeral session-context variable (not persisted)."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve an ephemeral session-context variable."""
        return self._context.get(key, default)

    def clear_context(self):
        """Wipe all ephemeral context variables."""
        self._context.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Union[str, Path]] = None):
        """Persist memory to disk.

        Args:
            path: Destination file path.  Uses *persist_path* if omitted.
        """
        target = Path(path) if path else self.persist_path
        if not target:
            self.logger.warning("[Memory] No path specified for save().")
            return

        data = {
            "history": [m.to_dict() for m in self._history],
            "long_term": self._long_term,
        }
        self._serializer.save(data, target)
        self.logger.debug(f"[Memory] Saved to {target}")

    def load(self, path: Optional[Union[str, Path]] = None):
        """Load memory from disk.

        Args:
            path: Source file path.  Uses *persist_path* if omitted.
        """
        target = Path(path) if path else self.persist_path
        if not target or not target.exists():
            self.logger.warning(f"[Memory] File not found: {target}")
            return

        data = self._serializer.load(target)
        self._history = deque(
            [Message.from_dict(m) for m in data.get("history", [])],
            maxlen=self.max_history,
        )
        self._long_term = data.get("long_term", {})
        self.logger.debug(f"[Memory] Loaded from {target}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auto_save(self):
        if self.persist_path:
            self.save()

    def __repr__(self):
        return (
            f"MemoryManager("
            f"history={len(self._history)}/{self.max_history}, "
            f"long_term_keys={len(self._long_term)})"
        )
