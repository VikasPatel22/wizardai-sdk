"""
WizardAI SDK
============
A powerful, all-in-one Python SDK for AI integration using the
Sagittarius Labs API (https://sagittarius-labs.pages.dev/).

Combines conversational AI, computer vision, speech I/O, memory
management, and a flexible plugin system into a single importable file.

Author  : WizardAI Contributors
Version : 1.0.0
License : MIT
PyPI    : pip install wizardai-sdk

Quick start::

    import wizardai

    wiz = wizardai.WizardAI(api_key="YOUR_API_KEY")
    wiz.start()
    print(wiz.ask("What is the speed of light?"))
    wiz.stop()

Get your API key at: https://sagittarius-labs.pages.dev/
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports (no third-party deps at module level)
# ---------------------------------------------------------------------------
import base64
import copy
import csv
import gzip
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pickle
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__version__   = "2.1.0"
__author__    = "WizardAI Contributors"
__license__   = "MIT"

# ---------------------------------------------------------------------------
# Sagittarius Labs API constants
# ---------------------------------------------------------------------------
_BASE_URL    = "https://sagittarius-labs.pages.dev"
_ENDPOINT    = f"{_BASE_URL}/api/chat"
_MODEL       = "sagittarius/deep-vl-r1-128b"
_SIGNUP_URL  = _BASE_URL
_ENV_KEY     = "WIZARDAI_API_KEY"


# =============================================================================
# Exceptions
# =============================================================================

class WizardAIError(Exception):
    """Base exception for all WizardAI errors."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.code = code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code})"


class APIError(WizardAIError):
    """Raised when an AI API call fails.

    Attributes:
        message : Human-readable error description.
        code    : HTTP status code.
    """

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message, code)


class AuthenticationError(APIError):
    """Raised when the API key is missing or invalid.

    Visit https://sagittarius-labs.pages.dev/ to obtain a valid API key.
    """

    def __init__(self, detail: str = ""):
        msg = (
            "Authentication failed — your API key is missing or invalid.\n"
            f"  Detail  : {detail}\n"
            f"  Fix     : Get or verify your key at {_SIGNUP_URL}\n"
            f"  Env var : export {_ENV_KEY}=<your_key>"
        )
        super().__init__(msg, code=401)


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded."""

    def __init__(self, retry_after: Optional[float] = None):
        super().__init__("Rate limit exceeded. Please wait before retrying.", code=429)
        self.retry_after = retry_after


class VisionError(WizardAIError):
    """Raised when a camera or image processing operation fails."""


class CameraNotFoundError(VisionError):
    """Raised when the requested camera device is not found."""

    def __init__(self, device_id: int = 0):
        super().__init__(f"Camera device {device_id} not found or unavailable.")
        self.device_id = device_id


class SpeechError(WizardAIError):
    """Raised when a speech recognition or TTS operation fails."""


class MicrophoneNotFoundError(SpeechError):
    """Raised when no microphone is detected."""

    def __init__(self):
        super().__init__("No microphone device found. Please check your audio input.")


class ConversationError(WizardAIError):
    """Raised when the conversation engine encounters an error."""


class PluginError(WizardAIError):
    """Raised when a plugin fails to load or execute."""

    def __init__(self, message: str, plugin_name: Optional[str] = None):
        super().__init__(message)
        self.plugin_name = plugin_name


class ConfigurationError(WizardAIError):
    """Raised when the SDK is misconfigured."""


# =============================================================================
# Logger
# =============================================================================

class Logger:
    """Configurable coloured logger for WizardAI components.

    Example::

        log = Logger("my_app", level="DEBUG")
        log.info("Started")
        log.warning("Low memory")
        log.error("Something went wrong")
    """

    _COLOURS = {
        "DEBUG":    "\033[94m",
        "INFO":     "\033[92m",
        "WARNING":  "\033[93m",
        "ERROR":    "\033[91m",
        "CRITICAL": "\033[95m",
        "RESET":    "\033[0m",
    }

    def __init__(
        self,
        name: str = "wizardai",
        level: str = "INFO",
        log_file: Optional[str] = None,
        coloured: bool = True,
    ):
        self.name = name
        self.coloured = coloured
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._build_formatter())
            self._logger.addHandler(handler)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(fh)

    # ------------------------------------------------------------------
    def _build_formatter(self) -> logging.Formatter:
        if self.coloured:
            _c = self._COLOURS

            class _CF(logging.Formatter):
                def format(self, record):  # noqa: A003
                    colour = _c.get(record.levelname, "")
                    reset  = _c["RESET"]
                    record.levelname = f"{colour}{record.levelname}{reset}"
                    return super().format(record)

            return _CF(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        return logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def debug(self, msg: str, *args, **kwargs):    self._logger.debug(msg, *args, **kwargs)
    def info(self, msg: str, *args, **kwargs):     self._logger.info(msg, *args, **kwargs)
    def warning(self, msg: str, *args, **kwargs):  self._logger.warning(msg, *args, **kwargs)
    def error(self, msg: str, *args, **kwargs):    self._logger.error(msg, *args, **kwargs)
    def critical(self, msg: str, *args, **kwargs): self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: str):
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# =============================================================================
# FileHelper
# =============================================================================

class FileHelper:
    """High-level file I/O helpers.

    Example::

        fh = FileHelper(base_dir="./data")
        fh.write_text("note.txt", "Hello!")
        content = fh.read_text("note.txt")
    """

    def __init__(self, base_dir: Union[str, Path] = "."):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def resolve(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        full = self.resolve(path)
        full.mkdir(parents=True, exist_ok=True)
        return full

    # ------------------------------------------------------------------
    def write_text(
        self,
        path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
        append: bool = False,
    ) -> Path:
        full = self.resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(full, mode, encoding=encoding) as fh:
            fh.write(content)
        return full

    def read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        with open(self.resolve(path), "r", encoding=encoding) as fh:
            return fh.read()

    def read_lines(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        strip: bool = True,
    ) -> List[str]:
        lines = self.read_text(path, encoding).splitlines()
        return [ln.strip() for ln in lines] if strip else lines

    # ------------------------------------------------------------------
    def write_json(self, path: Union[str, Path], data: Any, indent: int = 2) -> Path:
        full = self.resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
        return full

    def read_json(self, path: Union[str, Path]) -> Any:
        with open(self.resolve(path), "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    def write_csv(
        self,
        path: Union[str, Path],
        rows: List[Dict],
        fieldnames: Optional[List[str]] = None,
    ) -> Path:
        full = self.resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        if not fieldnames and rows:
            fieldnames = list(rows[0].keys())
        with open(full, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames or [])
            writer.writeheader()
            writer.writerows(rows)
        return full

    def read_csv(self, path: Union[str, Path]) -> List[Dict]:
        with open(self.resolve(path), "r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # ------------------------------------------------------------------
    def copy(self, src: Union[str, Path], dst: Union[str, Path]) -> Path:
        dst_path = self.resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(self.resolve(src)), str(dst_path))
        return dst_path

    def delete(self, path: Union[str, Path]) -> bool:
        full = self.resolve(path)
        if full.exists():
            full.unlink()
            return True
        return False

    def list_files(
        self,
        directory: Union[str, Path] = ".",
        pattern: str = "*",
        recursive: bool = False,
    ) -> List[Path]:
        full = self.resolve(directory)
        return list(full.rglob(pattern)) if recursive else list(full.glob(pattern))

    def timestamp_filename(self, name: str, ext: str = "") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ext if ext.startswith(".") or not ext else f".{ext}"
        return f"{name}_{ts}{suffix}"


# =============================================================================
# DataSerializer
# =============================================================================

class DataSerializer:
    """Serialize/deserialize data to JSON, Pickle, or gzip variants.

    Example::

        ds = DataSerializer()
        ds.save({"key": "val"}, "data.json")
        data = ds.load("data.json")
    """

    @staticmethod
    def _fmt(path: Union[str, Path]) -> str:
        name = str(path).lower()
        if name.endswith(".json.gz"):   return "json.gz"
        if name.endswith(".json"):      return "json"
        if name.endswith((".pkl.gz", ".pickle.gz")): return "pickle.gz"
        if name.endswith((".pkl", ".pickle")):        return "pickle"
        return "json"

    def save(
        self,
        data: Any,
        path: Union[str, Path],
        compress: bool = False,
        indent: int = 2,
    ) -> Path:
        p = Path(path)
        fmt = self._fmt(p)
        if compress and not fmt.endswith(".gz"):
            fmt += ".gz"
            p = Path(str(p) + ".gz")
        p.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=indent, ensure_ascii=False)
        elif fmt == "json.gz":
            with gzip.open(p, "wt", encoding="utf-8") as fh:
                json.dump(data, fh, indent=indent, ensure_ascii=False)
        elif fmt == "pickle":
            with open(p, "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt == "pickle.gz":
            with gzip.open(p, "wb") as fh:
                pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported format for: {path}")
        return p

    def load(self, path: Union[str, Path]) -> Any:
        p   = Path(path)
        fmt = self._fmt(p)
        if fmt == "json":
            with open(p, "r", encoding="utf-8") as fh:    return json.load(fh)
        if fmt == "json.gz":
            with gzip.open(p, "rt", encoding="utf-8") as fh: return json.load(fh)
        if fmt == "pickle":
            with open(p, "rb") as fh:                      return pickle.load(fh)
        if fmt == "pickle.gz":
            with gzip.open(p, "rb") as fh:                 return pickle.load(fh)
        raise ValueError(f"Unsupported format for: {path}")

    def to_json_string(self, data: Any, indent: int = 2) -> str:
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def from_json_string(self, text: str) -> Any:
        return json.loads(text)

    def iter_jsonl(self, path: Union[str, Path]) -> Iterator[Any]:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def write_jsonl(self, path: Union[str, Path], records: List[Any]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return p


# =============================================================================
# RateLimiter
# =============================================================================

class RateLimiter:
    """Token-bucket rate limiter.

    Example::

        limiter = RateLimiter(max_calls=10, period=60)
        for item in items:
            limiter.wait()
            process(item)
    """

    def __init__(self, max_calls: int = 60, period: float = 60.0):
        self.max_calls = max_calls
        self.period    = period
        self._timestamps: List[float] = []

    def wait(self):
        now = time.monotonic()
        self._timestamps = [t for t in self._timestamps if now - t < self.period]
        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.period - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._timestamps = self._timestamps[1:]
        self._timestamps.append(time.monotonic())

    def is_allowed(self) -> bool:
        now    = time.monotonic()
        active = [t for t in self._timestamps if now - t < self.period]
        return len(active) < self.max_calls

    # context-manager support
    def __enter__(self):
        self.wait()
        return self

    def __exit__(self, *_):
        pass


# =============================================================================
# Memory
# =============================================================================

class Message:
    """A single conversation message.

    Attributes:
        role      : 'user' | 'assistant' | 'system'
        content   : Message text.
        timestamp : Unix timestamp.
        metadata  : Arbitrary key-value pairs.
    """

    __slots__ = ("role", "content", "timestamp", "metadata")

    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.role      = role
        self.content   = content
        self.timestamp = time.time()
        self.metadata  = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "role":      self.role,
            "content":   self.content,
            "timestamp": self.timestamp,
            "metadata":  self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        msg           = cls(data["role"], data["content"], data.get("metadata", {}))
        msg.timestamp = data.get("timestamp", time.time())
        return msg

    def __repr__(self) -> str:
        preview = self.content[:40] + "…" if len(self.content) > 40 else self.content
        return f"Message(role={self.role!r}, content={preview!r})"


class MemoryManager:
    """Short-term conversation history and long-term key-value memory.

    Example::

        mem = MemoryManager(max_history=20)
        mem.add_message("user", "Hello!")
        mem.remember("user_name", "Alice")
        print(mem.recall("user_name"))   # → "Alice"
        mem.save("session.json")
    """

    def __init__(
        self,
        max_history: int = 50,
        persist_path: Optional[Union[str, Path]] = None,
        logger: Optional[Logger] = None,
    ):
        self.max_history  = max_history
        self.persist_path = Path(persist_path) if persist_path else None
        self.logger       = logger or Logger("MemoryManager")
        self._serializer  = DataSerializer()
        self._history: Deque[Message] = deque(maxlen=max_history)
        self._long_term:  Dict[str, Any] = {}
        self._context:    Dict[str, Any] = {}

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
        msgs = list(self._history)
        if role_filter:
            msgs = [m for m in msgs if m.role == role_filter]
        if n is not None:
            msgs = msgs[-n:]
        return msgs

    def get_history_as_dicts(self, n: Optional[int] = None) -> List[Dict]:
        return [m.to_dict() for m in self.get_history(n)]

    def get_messages_for_api(
        self,
        n: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        msgs = self.get_history(n)
        if not include_system:
            msgs = [m for m in msgs if m.role != "system"]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def clear_history(self):
        self._history.clear()
        self.logger.debug("[Memory] Short-term history cleared.")
        self._auto_save()

    def last_message(self, role: Optional[str] = None) -> Optional[Message]:
        msgs = list(self._history)
        if role:
            msgs = [m for m in msgs if m.role == role]
        return msgs[-1] if msgs else None

    def search_history(self, query: str, top_k: int = 5) -> List[Tuple[Message, float]]:
        query_words = set(query.lower().split())
        results: List[Tuple[Message, float]] = []
        for msg in self._history:
            words   = set(msg.content.lower().split())
            overlap = len(query_words & words)
            if overlap:
                results.append((msg, overlap / max(len(query_words), 1)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Long-term memory
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any):
        self._long_term[key] = value
        self.logger.debug(f"[Memory] Stored: {key!r}")
        self._auto_save()

    def recall(self, key: str, default: Any = None) -> Any:
        return self._long_term.get(key, default)

    def forget(self, key: str) -> bool:
        if key in self._long_term:
            del self._long_term[key]
            self._auto_save()
            return True
        return False

    def list_memories(self) -> List[str]:
        return list(self._long_term.keys())

    # ------------------------------------------------------------------
    # Ephemeral context (not persisted)
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: Any):
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def clear_context(self):
        self._context.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Union[str, Path]] = None):
        target = Path(path) if path else self.persist_path
        if not target:
            self.logger.warning("[Memory] No path specified for save().")
            return
        data = {
            "history":    [m.to_dict() for m in self._history],
            "long_term":  self._long_term,
        }
        self._serializer.save(data, target)
        self.logger.debug(f"[Memory] Saved to {target}")

    def load(self, path: Optional[Union[str, Path]] = None):
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

    def _auto_save(self):
        if self.persist_path:
            self.save()

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"history={len(self._history)}/{self.max_history}, "
            f"long_term_keys={len(self._long_term)})"
        )


# =============================================================================
# Conversation
# =============================================================================

def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Convert a WizardAI pattern string to a compiled regex.

    Wildcards:
        ``*``      → one or more words (greedy).
        ``?``      → exactly one word.
        ``{name}`` → named capture group.
    """
    tokens = re.split(r"(\*|\?|\{[^}]+\})", pattern)
    parts  = []
    for token in tokens:
        if token == "*":
            parts.append(r"(.+)")
        elif token == "?":
            parts.append(r"(\S+)")
        elif token.startswith("{") and token.endswith("}"):
            name = token[1:-1]
            parts.append(f"(?P<{name}>.+?)")
        else:
            parts.append(re.escape(token))
    return re.compile(r"^\s*" + "".join(parts) + r"\s*$", re.IGNORECASE)


@dataclass
class Pattern:
    """A single conversation rule.

    Attributes:
        pattern  : Input pattern string (supports wildcards ``*``, ``?``, ``{name}``).
        template : Response — str, callable, or list of alternatives.
        priority : Higher values match first.
        context  : Optional context key required for this rule to fire.
        tags     : Arbitrary labels.
    """

    pattern:  str
    template: Union[str, Callable[..., str], List[str]]
    priority: int           = 0
    context:  Optional[str] = None
    tags:     List[str]     = field(default_factory=list)
    _regex:   Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._regex = None

    def compile(self) -> re.Pattern:
        if self._regex is None:
            self._regex = _pattern_to_regex(self.pattern)
        return self._regex


class ConversationAgent:
    """AIML-style rule-based chat engine with wildcards, context, and memory.

    Example::

        agent = ConversationAgent(name="WizardBot")
        agent.add_pattern("hello", "Hello! How can I help?")
        agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")
        print(agent.respond("hello"))
    """

    def __init__(
        self,
        name: str = "WizardBot",
        fallback: str = "I'm not sure how to respond to that.",
        memory: Optional[MemoryManager] = None,
        logger: Optional[Logger] = None,
        case_sensitive: bool = False,
    ):
        self.name           = name
        self.fallback       = fallback
        self.memory         = memory or MemoryManager(max_history=100)
        self.logger         = logger or Logger("ConversationAgent")
        self.case_sensitive = case_sensitive

        self._patterns:       List[Pattern]             = []
        self._active_context: Optional[str]             = None
        self._plugins:        Dict[str, Callable]       = {}
        self._preprocessors:  List[Callable[[str], str]] = []
        self._postprocessors: List[Callable[[str], str]] = []

        self._register_defaults()

    # ------------------------------------------------------------------
    # Pattern management
    # ------------------------------------------------------------------

    def add_pattern(
        self,
        pattern: str,
        template: Union[str, Callable, List[str]],
        priority: int = 0,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Pattern:
        p = Pattern(
            pattern=pattern,
            template=template,
            priority=priority,
            context=context,
            tags=tags or [],
        )
        p.compile()
        self._patterns.append(p)
        self._patterns.sort(key=lambda x: x.priority, reverse=True)
        self.logger.debug(f"[Agent] Pattern: {pattern!r} (priority={priority})")
        return p

    def add_pattern_obj(self, p: Pattern) -> Pattern:
        """Register a pre-built :class:`Pattern` object."""
        p.compile()
        self._patterns.append(p)
        self._patterns.sort(key=lambda x: x.priority, reverse=True)
        return p

    def remove_pattern(self, pattern_str: str) -> int:
        before = len(self._patterns)
        self._patterns = [p for p in self._patterns if p.pattern != pattern_str]
        return before - len(self._patterns)

    def clear_patterns(self):
        self._patterns.clear()

    def load_patterns_from_dict(self, rules: Dict[str, Any]):
        for pat, tmpl in rules.items():
            self.add_pattern(pat, tmpl)
        self.logger.info(f"[Agent] Loaded {len(rules)} pattern(s).")

    def load_patterns_from_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        self.load_patterns_from_dict(rules)

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def respond(self, user_input: str) -> str:
        # 1. Check for plugin invocation (!plugin_name args)
        plugin_resp = self._dispatch_plugin(user_input)
        if plugin_resp is not None:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", plugin_resp)
            return plugin_resp

        processed = self._preprocess(user_input)
        response, matched = self._match(processed)
        response = self._postprocess(response)

        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", response)
        self.logger.debug(
            f"[Agent] Input={processed!r}, Pattern={matched!r}, "
            f"Response={response[:60]!r}"
        )
        return response

    def _match(self, text: str) -> Tuple[str, Optional[str]]:
        for pattern in self._patterns:
            if pattern.context and pattern.context != self._active_context:
                continue
            regex = pattern.compile()
            m     = regex.match(text)
            if m:
                resp = self._render_template(pattern.template, m)
                if pattern.context:
                    self._active_context = pattern.context
                return resp, pattern.pattern
        return self.fallback, None

    def _render_template(
        self,
        template: Union[str, Callable, List[str]],
        match: re.Match,
    ) -> str:
        import random

        if callable(template):
            try:
                return str(template())
            except Exception as exc:
                self.logger.error(f"Template callable error: {exc}")
                return self.fallback

        if isinstance(template, list):
            template = random.choice(template)

        groups = match.groups()
        named  = match.groupdict()

        if "{wildcard}" in template and groups:
            template = template.replace("{wildcard}", groups[0])

        for i, group in enumerate(groups):
            template = template.replace(f"{{{i}}}", group or "")

        for name, value in named.items():
            template = template.replace(f"{{{name}}}", value or "")

        # {memory:key} substitution
        template = re.sub(
            r"\{memory:([^}]+)\}",
            lambda m: str(self.memory.recall(m.group(1), "")),
            template,
        )
        return template

    # ------------------------------------------------------------------
    # Pre/post processors
    # ------------------------------------------------------------------

    def add_preprocessor(self, fn: Callable[[str], str]):
        self._preprocessors.append(fn)

    def add_postprocessor(self, fn: Callable[[str], str]):
        self._postprocessors.append(fn)

    def _preprocess(self, text: str) -> str:
        result = text
        for fn in self._preprocessors:
            try:
                result = fn(result)
            except Exception as exc:
                self.logger.warning(f"Preprocessor error: {exc}")
        return result

    def _postprocess(self, text: str) -> str:
        result = text
        for fn in self._postprocessors:
            try:
                result = fn(result)
            except Exception as exc:
                self.logger.warning(f"Postprocessor error: {exc}")
        return result

    # ------------------------------------------------------------------
    # Inline plugins
    # ------------------------------------------------------------------

    def register_plugin(self, name: str, handler: Callable):
        """Register a ``!name`` invocable plugin handler."""
        self._plugins[name.lower()] = handler
        self.logger.info(f"[Agent] Inline plugin registered: {name!r}")

    def _dispatch_plugin(self, text: str) -> Optional[str]:
        if not text.startswith("!"):
            return None
        parts = text[1:].split(None, 1)
        if not parts:
            return None
        name    = parts[0].lower()
        args    = parts[1] if len(parts) > 1 else ""
        handler = self._plugins.get(name)
        if handler:
            try:
                return str(handler(args))
            except Exception as exc:
                return f"Plugin '{name}' error: {exc}"
        return None

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def set_context(self, context: str):
        self._active_context = context

    def clear_context(self):
        self._active_context = None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def _register_defaults(self):
        defaults: Dict[str, Any] = {
            "hello":              ["Hello!", "Hi there!", "Hey! How can I help you?"],
            "hi":                 ["Hi!", "Hello!", "Hey!"],
            "how are you":        ["I'm doing great, thanks!", "Ready to assist you!"],
            "what is your name":  f"I'm {self.name}, your AI assistant.",
            "what can you do":    "I can answer questions, hold conversations, and more!",
            "goodbye":            ["Goodbye! Have a great day!", "See you later!"],
            "bye":                ["Bye!", "Goodbye!", "Take care!"],
            "thank you":          ["You're welcome!", "Happy to help!", "Anytime!"],
            "thanks":             ["No problem!", "Glad I could help!"],
        }
        for pat, tmpl in defaults.items():
            self.add_pattern(pat, tmpl, priority=-10)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_patterns(self, tag: Optional[str] = None) -> List[Pattern]:
        if tag:
            return [p for p in self._patterns if tag in p.tags]
        return list(self._patterns)

    def get_history(self, n: int = 10) -> List[Dict]:
        return self.memory.get_history_as_dicts(n)

    def reset(self):
        self.memory.clear_history()
        self._active_context = None
        self.logger.info("[Agent] Conversation reset.")

    def __repr__(self) -> str:
        return (
            f"ConversationAgent(name={self.name!r}, "
            f"patterns={len(self._patterns)}, "
            f"context={self._active_context!r})"
        )


# =============================================================================
# Plugin System
# =============================================================================

class PluginBase(ABC):
    """Abstract base class for all WizardAI plugins.

    Example::

        class JokePlugin(PluginBase):
            name        = "jokes"
            description = "Tells jokes."
            version     = "1.0.0"
            triggers    = ["tell me a joke", "joke"]

            def on_message(self, text, context):
                return "Why do Python devs wear glasses? They can't C!"

        manager = PluginManager()
        manager.register(JokePlugin)
        print(manager.dispatch("tell me a joke"))
    """

    name:        str       = "unnamed_plugin"
    description: str       = ""
    version:     str       = "0.0.1"
    author:      str       = ""
    triggers:    List[str] = []

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
    ):
        self.config   = config or {}
        self.logger   = logger or Logger(f"Plugin:{self.name}")
        self._enabled = True
        self.setup()

    def setup(self):
        """Called once after __init__. Override to initialise resources."""

    def teardown(self):
        """Called when unregistered. Override to clean up."""

    @abstractmethod
    def on_message(self, text: str, context: Dict[str, Any]) -> Optional[str]:
        """Process text and return a response, or None to pass through."""

    def on_start(self):
        """Called when the WizardAI session starts."""

    def on_stop(self):
        """Called when the WizardAI session ends."""

    def enable(self):
        self._enabled = True
        self.logger.info(f"Plugin '{self.name}' enabled.")

    def disable(self):
        self._enabled = False
        self.logger.info(f"Plugin '{self.name}' disabled.")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"


class PluginManager:
    """Manages lifecycle, registration, and dispatch of WizardAI plugins.

    Example::

        manager = PluginManager()
        manager.register(JokePlugin)
        response = manager.dispatch("tell me a joke", context={})
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger   = logger or Logger("PluginManager")
        self._plugins: Dict[str, PluginBase] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        plugin_cls: Type[PluginBase],
        config: Optional[Dict[str, Any]] = None,
        name_override: Optional[str] = None,
    ) -> PluginBase:
        if not (inspect.isclass(plugin_cls) and issubclass(plugin_cls, PluginBase)):
            raise PluginError(f"{plugin_cls!r} is not a PluginBase subclass.")

        name = name_override or plugin_cls.name
        if name in self._plugins:
            raise PluginError(
                f"Plugin '{name}' is already registered. Use name_override.",
                plugin_name=name,
            )
        try:
            instance = plugin_cls(config=config, logger=self.logger)
        except Exception as exc:
            raise PluginError(
                f"Failed to instantiate plugin '{name}': {exc}",
                plugin_name=name,
            ) from exc

        self._plugins[name] = instance
        self.logger.info(f"Plugin registered: {name!r} v{instance.version}")
        return instance

    def unregister(self, name: str) -> bool:
        plugin = self._plugins.pop(name, None)
        if plugin:
            try:
                plugin.teardown()
            except Exception as exc:
                self.logger.warning(f"Plugin teardown error ({name}): {exc}")
            self.logger.info(f"Plugin unregistered: {name!r}")
            return True
        return False

    def get(self, name: str) -> Optional[PluginBase]:
        return self._plugins.get(name)

    def list_plugins(self, enabled_only: bool = False) -> List[PluginBase]:
        plugins = list(self._plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p.is_enabled]
        return plugins

    # ------------------------------------------------------------------
    def dispatch(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        ctx = context or {}
        for plugin in self._plugins.values():
            if not plugin.is_enabled:
                continue
            try:
                result = plugin.on_message(text, ctx)
                if result is not None:
                    self.logger.debug(f"[Dispatch] '{plugin.name}' handled: {text[:40]!r}")
                    return result
            except Exception as exc:
                self.logger.error(f"[Dispatch] Plugin '{plugin.name}' error: {exc}")
        return None

    def dispatch_all(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str]]:
        ctx     = context or {}
        results: List[Tuple[str, str]] = []
        for name, plugin in self._plugins.items():
            if not plugin.is_enabled:
                continue
            try:
                result = plugin.on_message(text, ctx)
                if result is not None:
                    results.append((name, result))
            except Exception as exc:
                self.logger.error(f"[DispatchAll] Plugin '{name}' error: {exc}")
        return results

    # ------------------------------------------------------------------
    def load_from_file(
        self,
        path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> PluginBase:
        p = Path(path).resolve()
        if not p.exists():
            raise PluginError(f"Plugin file not found: {p}")

        module_name = f"_wizardai_plugin_{p.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(p))
        if spec is None or spec.loader is None:
            raise PluginError(f"Could not load spec from: {p}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            raise PluginError(f"Error executing plugin file {p}: {exc}") from exc

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, PluginBase)
                and obj is not PluginBase
            ):
                return self.register(obj, config=config)

        raise PluginError(
            f"No PluginBase subclass found in {p}. "
            "Ensure your class inherits from PluginBase."
        )

    def load_from_directory(
        self,
        directory: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[PluginBase]:
        d = Path(directory).resolve()
        if not d.is_dir():
            raise PluginError(f"Not a directory: {d}")
        loaded = []
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                loaded.append(self.load_from_file(py_file, config=config))
            except PluginError as exc:
                self.logger.warning(f"Skipping {py_file.name}: {exc}")
        self.logger.info(f"Loaded {len(loaded)} plugin(s) from {d}")
        return loaded

    # ------------------------------------------------------------------
    def start_all(self):
        for plugin in self._plugins.values():
            if plugin.is_enabled:
                try:
                    plugin.on_start()
                except Exception as exc:
                    self.logger.error(f"Plugin start error ({plugin.name}): {exc}")

    def stop_all(self):
        for plugin in self._plugins.values():
            try:
                plugin.on_stop()
            except Exception as exc:
                self.logger.error(f"Plugin stop error ({plugin.name}): {exc}")

    def __len__(self):
        return len(self._plugins)

    def __repr__(self) -> str:
        return f"PluginManager(plugins={list(self._plugins.keys())})"


# =============================================================================
# Vision Module  (optional — requires opencv-python)
# =============================================================================

# Type alias; avoids importing numpy at module level
Frame = Any   # numpy.ndarray (BGR)


class VisionModule:
    """Real-time camera access and image processing via OpenCV.

    Requires: ``pip install opencv-python``

    Example::

        cam = VisionModule(device_id=0)
        cam.open()
        frame = cam.capture_frame()
        cam.save_frame(frame, "snap.jpg")
        cam.close()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        logger: Optional[Logger] = None,
    ):
        self.device_id  = device_id
        self.width      = width
        self.height     = height
        self.fps        = fps
        self.logger     = logger or Logger("VisionModule")

        self._cap            = None
        self._stream_thread: Optional[threading.Thread] = None
        self._streaming      = threading.Event()
        self._face_cascade   = None
        self._frame_callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    def open(self):
        try:
            import cv2
        except ImportError:
            raise VisionError(
                "OpenCV is required. pip install opencv-python"
            )
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise CameraNotFoundError(self.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
        self.logger.info(
            f"Camera {self.device_id} opened: {self.width}x{self.height}@{self.fps}fps"
        )

    def close(self):
        self.stop_stream()
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self.logger.info(f"Camera {self.device_id} released.")
        self._cap = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    def capture_frame(self) -> Frame:
        if not self.is_open():
            raise VisionError("Camera is not open. Call open() first.")
        ret, frame = self._cap.read()
        if not ret:
            raise VisionError("Failed to capture frame.")
        return frame

    def capture_frames(self, n: int, delay: float = 0.0) -> List[Frame]:
        frames = []
        for _ in range(n):
            frames.append(self.capture_frame())
            if delay > 0:
                time.sleep(delay)
        return frames

    # ------------------------------------------------------------------
    def save_frame(
        self,
        frame: Frame,
        path: Union[str, Path],
        quality: int = 95,
    ) -> Path:
        try:
            import cv2
        except ImportError:
            raise VisionError("OpenCV is required. pip install opencv-python")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        params = []
        if p.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        cv2.imwrite(str(p), frame, params)
        self.logger.debug(f"Frame saved to {p}")
        return p

    def load_image(self, path: Union[str, Path]) -> Frame:
        try:
            import cv2
        except ImportError:
            raise VisionError("OpenCV is required. pip install opencv-python")
        frame = cv2.imread(str(path))
        if frame is None:
            raise VisionError(f"Could not load image from: {path}")
        return frame

    # ------------------------------------------------------------------
    def resize_frame(self, frame: Frame, width: int, height: int) -> Frame:
        import cv2
        return cv2.resize(frame, (width, height))

    def to_grayscale(self, frame: Frame) -> Frame:
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def to_rgb(self, frame: Frame) -> Frame:
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def flip(self, frame: Frame, axis: int = 1) -> Frame:
        import cv2
        return cv2.flip(frame, axis)

    def draw_rectangle(
        self,
        frame: Frame,
        x: int, y: int, w: int, h: int,
        colour: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> Frame:
        import cv2
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, thickness)
        return frame

    def draw_text(
        self,
        frame: Frame,
        text: str,
        x: int, y: int,
        font_scale: float = 0.7,
        colour: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> Frame:
        import cv2
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness)
        return frame

    def encode_to_base64(self, frame: Frame, ext: str = ".jpg") -> str:
        import cv2
        ret, buf = cv2.imencode(ext, frame)
        if not ret:
            raise VisionError("Frame encoding failed.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    def detect_faces(
        self,
        frame: Frame,
        scale_factor: float = 1.1,
        min_neighbours: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[Dict]:
        try:
            import cv2
        except ImportError:
            raise VisionError("OpenCV is required. pip install opencv-python")

        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise VisionError("Could not load Haar cascade for face detection.")

        gray = self.to_grayscale(frame) if len(frame.shape) == 3 else frame
        detections = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbours,
            minSize=min_size,
        )
        return [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for x, y, w, h in (detections if len(detections) else [])
        ]

    def annotate_faces(self, frame: Frame) -> Tuple[Frame, List[Dict]]:
        faces     = self.detect_faces(frame)
        annotated = copy.deepcopy(frame)
        for face in faces:
            self.draw_rectangle(annotated, face["x"], face["y"], face["w"], face["h"])
            self.draw_text(annotated, "Face", face["x"], face["y"] - 10)
        return annotated, faces

    # ------------------------------------------------------------------
    def add_frame_callback(self, callback: Callable[[Frame], None]):
        self._frame_callbacks.append(callback)

    def start_stream(
        self,
        callback: Optional[Callable[[Frame], None]] = None,
        show_preview: bool = False,
    ):
        if self._streaming.is_set():
            self.logger.warning("Stream already running.")
            return
        if not self.is_open():
            self.open()
        if callback:
            self.add_frame_callback(callback)
        self._streaming.set()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(show_preview,),
            daemon=True,
            name="wizardai-vision-stream",
        )
        self._stream_thread.start()
        self.logger.info("Vision stream started.")

    def stop_stream(self):
        if not self._streaming.is_set():
            return
        self._streaming.clear()
        if self._stream_thread:
            self._stream_thread.join(timeout=3.0)
        self.logger.info("Vision stream stopped.")

    def _stream_loop(self, show_preview: bool):
        try:
            import cv2
        except ImportError:
            self.logger.error("OpenCV is required for streaming.")
            return

        interval = 1.0 / self.fps if self.fps > 0 else 0
        while self._streaming.is_set():
            t_start = time.monotonic()
            try:
                frame = self.capture_frame()
                for cb in self._frame_callbacks:
                    try:
                        cb(frame)
                    except Exception as exc:
                        self.logger.error(f"Frame callback error: {exc}")
                if show_preview:
                    cv2.imshow("WizardAI Vision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._streaming.clear()
                        break
            except VisionError as exc:
                self.logger.error(f"Vision error in stream loop: {exc}")
                break
            elapsed    = time.monotonic() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if show_preview:
            import cv2
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self) -> str:
        status    = "open" if self.is_open() else "closed"
        streaming = "streaming" if self._streaming.is_set() else "idle"
        return f"VisionModule(device={self.device_id}, {status}, {streaming})"


# =============================================================================
# Speech Module  (optional — requires SpeechRecognition / pyttsx3 / etc.)
# =============================================================================

class SpeechModule:
    """Speech recognition (STT) and text-to-speech (TTS).

    STT backends : 'google' | 'sphinx' | 'whisper'
    TTS backends : 'pyttsx3' | 'gtts' | 'elevenlabs'

    Example::

        speech = SpeechModule(stt_backend="google", tts_backend="pyttsx3")
        text   = speech.listen(timeout=5)
        speech.say("You said: " + text)
    """

    def __init__(
        self,
        stt_backend: str = "google",
        tts_backend: str = "pyttsx3",
        language: str = "en-US",
        tts_rate: int = 150,
        tts_volume: float = 1.0,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: Optional[str] = None,
        logger: Optional[Logger] = None,
    ):
        self.stt_backend          = stt_backend.lower()
        self.tts_backend          = tts_backend.lower()
        self.language             = language
        self.tts_rate             = tts_rate
        self.tts_volume           = tts_volume
        self.elevenlabs_api_key   = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id  = elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
        self.logger               = logger or Logger("SpeechModule")

        self._recogniser          = None
        self._tts_engine          = None
        self._whisper_model       = None
        self._listening           = threading.Event()
        self._continuous_callbacks: List[Callable[[str], None]] = []

        self._init_tts()
        self.logger.info(
            f"SpeechModule: STT={self.stt_backend}, TTS={self.tts_backend}, lang={self.language}"
        )

    # ------------------------------------------------------------------
    def _get_recogniser(self):
        if self._recogniser is None:
            try:
                import speech_recognition as sr
                self._recogniser = sr.Recognizer()
            except ImportError:
                raise SpeechError(
                    "SpeechRecognition is required. pip install SpeechRecognition"
                )
        return self._recogniser

    def _init_tts(self):
        if self.tts_backend == "pyttsx3":
            try:
                import pyttsx3
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate",   self.tts_rate)
                self._tts_engine.setProperty("volume", self.tts_volume)
            except ImportError:
                self.logger.warning("pyttsx3 not installed. pip install pyttsx3")
            except Exception as exc:
                self.logger.warning(f"pyttsx3 init error: {exc}")

    # ------------------------------------------------------------------
    def listen(
        self,
        timeout: Optional[float] = 5.0,
        phrase_time_limit: Optional[float] = 15.0,
        adjust_noise: bool = True,
        device_index: Optional[int] = None,
    ) -> str:
        try:
            import speech_recognition as sr
        except ImportError:
            raise SpeechError("SpeechRecognition required. pip install SpeechRecognition")

        recogniser = self._get_recogniser()
        try:
            mic = sr.Microphone(device_index=device_index)
        except (AttributeError, OSError):
            raise MicrophoneNotFoundError()

        with mic as source:
            if adjust_noise:
                recogniser.adjust_for_ambient_noise(source, duration=0.5)
            self.logger.debug("Listening…")
            try:
                audio = recogniser.listen(source, timeout=timeout,
                                          phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                raise SpeechError("No speech detected within timeout.")

        return self._transcribe(audio)

    def transcribe_file(self, path: Union[str, Path]) -> str:
        try:
            import speech_recognition as sr
        except ImportError:
            raise SpeechError("SpeechRecognition required. pip install SpeechRecognition")
        recogniser = self._get_recogniser()
        with sr.AudioFile(str(path)) as source:
            audio = recogniser.record(source)
        return self._transcribe(audio)

    def _transcribe(self, audio) -> str:
        import speech_recognition as sr
        recogniser = self._get_recogniser()
        try:
            if self.stt_backend == "google":
                return recogniser.recognize_google(audio, language=self.language)
            elif self.stt_backend == "sphinx":
                return recogniser.recognize_sphinx(audio)
            elif self.stt_backend == "whisper":
                return self._transcribe_whisper(audio)
            else:
                raise SpeechError(f"Unknown STT backend: {self.stt_backend!r}")
        except sr.UnknownValueError:
            raise SpeechError("Speech was unintelligible.")
        except sr.RequestError as exc:
            raise SpeechError(f"STT API request failed: {exc}")

    def _transcribe_whisper(self, audio) -> str:
        try:
            import whisper
            import numpy as np
        except ImportError:
            raise SpeechError(
                "Whisper STT requires openai-whisper and numpy. "
                "pip install openai-whisper numpy"
            )
        if self._whisper_model is None:
            self.logger.info("Loading Whisper model…")
            self._whisper_model = whisper.load_model("base")
        raw      = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        result   = self._whisper_model.transcribe(audio_np, language=self.language[:2])
        return result.get("text", "").strip()

    # ------------------------------------------------------------------
    def say(self, text: str, blocking: bool = True) -> Optional[str]:
        self.logger.debug(f"[TTS] {text[:60]!r}{'…' if len(text) > 60 else ''}")
        if self.tts_backend == "pyttsx3":
            self._say_pyttsx3(text, blocking)
        elif self.tts_backend == "gtts":
            return self._say_gtts(text)
        elif self.tts_backend == "elevenlabs":
            return self._say_elevenlabs(text)
        else:
            raise SpeechError(f"Unknown TTS backend: {self.tts_backend!r}")
        return None

    def synthesise_to_file(self, text: str, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.tts_backend == "pyttsx3":
            if self._tts_engine is None:
                raise SpeechError("pyttsx3 engine not initialised.")
            self._tts_engine.save_to_file(text, str(p))
            self._tts_engine.runAndWait()
        elif self.tts_backend == "gtts":
            try:
                from gtts import gTTS
            except ImportError:
                raise SpeechError("gTTS required. pip install gtts")
            gTTS(text=text, lang=self.language[:2]).save(str(p))
        elif self.tts_backend == "elevenlabs":
            p.write_bytes(self._elevenlabs_synthesise(text))
        else:
            raise SpeechError(f"Unknown TTS backend: {self.tts_backend!r}")
        return p

    def _say_pyttsx3(self, text: str, blocking: bool):
        if self._tts_engine is None:
            raise SpeechError("pyttsx3 not available. pip install pyttsx3")
        self._tts_engine.say(text)
        if blocking:
            self._tts_engine.runAndWait()

    def _say_gtts(self, text: str) -> str:
        try:
            from gtts import gTTS
        except ImportError:
            raise SpeechError("gTTS required. pip install gtts")
        try:
            import pygame
        except ImportError:
            pygame = None  # type: ignore[assignment]

        tts = gTTS(text=text, lang=self.language[:2], slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)
        if pygame:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
            except Exception as exc:
                self.logger.warning(f"pygame playback failed: {exc}")
        return tmp_path

    def _say_elevenlabs(self, text: str) -> str:
        audio_bytes = self._elevenlabs_synthesise(text)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except ImportError:
            self.logger.warning("pygame not installed; audio saved but not played.")
        return tmp_path

    def _elevenlabs_synthesise(self, text: str) -> bytes:
        if not self.elevenlabs_api_key:
            raise SpeechError(
                "ElevenLabs API key is required. "
                "Set env var ELEVENLABS_API_KEY or pass elevenlabs_api_key=."
            )
        try:
            import requests as _req
        except ImportError:
            raise SpeechError("requests required. pip install requests")
        url     = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
        headers = {"xi-api-key": self.elevenlabs_api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        r = _req.post(url, headers=headers, json=payload, timeout=30)
        if not r.ok:
            raise SpeechError(f"ElevenLabs error {r.status_code}: {r.text}")
        return r.content

    # ------------------------------------------------------------------
    def add_listener(self, callback: Callable[[str], None]):
        self._continuous_callbacks.append(callback)

    def start_continuous_listening(
        self,
        callback: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
        phrase_time_limit: float = 10.0,
    ):
        if self._listening.is_set():
            self.logger.warning("Continuous listening already active.")
            return
        if callback:
            self.add_listener(callback)
        self._listening.set()
        t = threading.Thread(
            target=self._continuous_loop,
            args=(timeout, phrase_time_limit),
            daemon=True,
            name="wizardai-speech-listen",
        )
        t.start()
        self.logger.info("Continuous listening started.")

    def stop_continuous_listening(self):
        self._listening.clear()
        self.logger.info("Continuous listening stopped.")

    def _continuous_loop(self, timeout, phrase_time_limit):
        while self._listening.is_set():
            try:
                text = self.listen(timeout=timeout, phrase_time_limit=phrase_time_limit)
                if text:
                    for cb in self._continuous_callbacks:
                        try:
                            cb(text)
                        except Exception as exc:
                            self.logger.error(f"Speech callback error: {exc}")
            except SpeechError as exc:
                self.logger.debug(f"[Listen loop] {exc}")
            except Exception as exc:
                self.logger.error(f"Unexpected error in listen loop: {exc}")

    # ------------------------------------------------------------------
    def list_microphones(self) -> List[Dict]:
        try:
            import speech_recognition as sr
        except ImportError:
            raise SpeechError("SpeechRecognition required. pip install SpeechRecognition")
        return [
            {"index": i, "name": name}
            for i, name in enumerate(sr.Microphone.list_microphone_names())
        ]

    def set_tts_rate(self, rate: int):
        self.tts_rate = rate
        if self._tts_engine:
            self._tts_engine.setProperty("rate", rate)

    def set_tts_volume(self, volume: float):
        self.tts_volume = max(0.0, min(1.0, volume))
        if self._tts_engine:
            self._tts_engine.setProperty("volume", self.tts_volume)

    def set_tts_voice(self, voice_id: str):
        if self._tts_engine:
            self._tts_engine.setProperty("voice", voice_id)

    def list_voices(self) -> List[Dict]:
        if not self._tts_engine:
            return []
        return [
            {"id": v.id, "name": v.name, "languages": v.languages}
            for v in self._tts_engine.getProperty("voices")
        ]

    def __repr__(self) -> str:
        return (
            f"SpeechModule(stt={self.stt_backend!r}, "
            f"tts={self.tts_backend!r}, lang={self.language!r})"
        )


# =============================================================================
# AI Response dataclass
# =============================================================================

@dataclass
class AIResponse:
    """Structured response from the Sagittarius Labs API.

    Attributes:
        text       : Generated text content.
        model      : Model identifier used.
        usage      : Token usage stats.
        raw        : Raw API response dict.
        latency_ms : Round-trip latency in milliseconds.
    """
    text:       str
    model:      str             = _MODEL
    usage:      Dict[str, int]  = field(default_factory=dict)
    raw:        Dict[str, Any]  = field(default_factory=dict)
    latency_ms: float           = 0.0

    def __str__(self) -> str:
        return self.text


# =============================================================================
# AI Client  (Sagittarius Labs API only)
# =============================================================================

class AIClient:
    """Client for the Sagittarius Labs AI API.

    All requests go to ``https://sagittarius-labs.pages.dev/api/chat``
    using the ``sagittarius/deep-vl-r1-128b`` model.

    Authentication errors direct users to https://sagittarius-labs.pages.dev/
    to obtain or verify their API key.

    Example::

        client = AIClient(api_key="YOUR_KEY")
        response = client.chat([{"role": "user", "content": "Hello!"}])
        print(response.text)

        # Streaming
        for chunk in client.chat_stream([{"role": "user", "content": "Tell me a story"}]):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _MODEL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        rate_limit_calls: int = 60,
        rate_limit_period: float = 60.0,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            api_key            : Your Sagittarius Labs API key.
                                 Falls back to ``WIZARDAI_API_KEY`` env var.
                                 Obtain one at https://sagittarius-labs.pages.dev/
            model              : Model to use (default: sagittarius/deep-vl-r1-128b).
            max_retries        : Retry attempts on transient errors.
            retry_delay        : Initial delay in seconds (doubles each attempt).
            timeout            : HTTP timeout in seconds.
            rate_limit_calls   : Max API calls per window.
            rate_limit_period  : Rate-limit window in seconds.
            logger             : Optional Logger instance.
        """
        self.api_key     = api_key or os.environ.get(_ENV_KEY, "")
        self.model       = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout     = timeout
        self.logger      = logger or Logger("AIClient")
        self._rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)

        if not self.api_key:
            self.logger.warning(
                f"No API key found. Set env var {_ENV_KEY}=<your_key> "
                f"or obtain one at {_SIGNUP_URL}"
            )

        self.logger.info(f"AIClient ready | model={self.model}")

    # ------------------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _build_messages(
        messages: List[Dict[str, str]],
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        if system_prompt:
            return [{"role": "system", "content": system_prompt}] + list(messages)
        return list(messages)

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AIResponse:
        """Send a multi-turn chat request (non-streaming).

        Args:
            messages      : List of ``{"role": ..., "content": ...}`` dicts.
            model         : Override the default model.
            max_tokens    : Maximum tokens to generate.
            temperature   : Sampling temperature (0 = deterministic).
            system_prompt : Prepend a system message.

        Returns:
            :class:`AIResponse`

        Raises:
            AuthenticationError : Invalid or missing API key.
            RateLimitError      : Rate limit exceeded.
            APIError            : Other API failures.
        """
        _messages = self._build_messages(messages, system_prompt)
        return self._with_retry(
            self._call,
            messages=_messages,
            model=model or self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Send a chat request and stream the response token by token.

        Yields:
            str chunks as they arrive.

        Example::

            for chunk in client.chat_stream([{"role": "user", "content": "Hi!"}]):
                print(chunk, end="", flush=True)
        """
        _messages = self._build_messages(messages, system_prompt)
        self._rate_limiter.wait()
        yield from self._stream(
            messages=_messages,
            model=model or self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AIResponse:
        """Single-turn convenience wrapper around :meth:`chat`."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def complete_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Single-turn streaming convenience wrapper."""
        yield from self.chat_stream(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def set_api_key(self, api_key: str):
        """Update the API key at runtime."""
        self.api_key = api_key
        self.logger.info("API key updated.")

    def set_model(self, model: str):
        """Change the default model."""
        self.model = model
        self.logger.info(f"Model changed to: {model}")

    # ------------------------------------------------------------------
    # Internal: retry logic
    # ------------------------------------------------------------------

    def _with_retry(self, fn: Callable, **kwargs) -> AIResponse:
        self._rate_limiter.wait()
        last_error: Optional[Exception] = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 2):
            try:
                t_start  = time.monotonic()
                response = fn(**kwargs)
                response.latency_ms = (time.monotonic() - t_start) * 1000
                return response

            except AuthenticationError:
                raise  # never retry auth failures

            except RateLimitError as exc:
                last_error = exc
                wait = exc.retry_after or delay
                self.logger.warning(f"Rate limited. Waiting {wait:.1f}s (attempt {attempt})…")
                time.sleep(wait)
                delay *= 2

            except APIError as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    self.logger.warning(
                        f"API error (attempt {attempt}/{self.max_retries}): "
                        f"{exc.message}. Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

            except Exception as exc:
                last_error = APIError(str(exc))
                if attempt <= self.max_retries:
                    self.logger.warning(
                        f"Unexpected error (attempt {attempt}): {exc}. "
                        f"Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise last_error from exc

        raise last_error or APIError("Max retries exceeded")

    # ------------------------------------------------------------------
    # Internal: HTTP calls
    # ------------------------------------------------------------------

    def _call(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> AIResponse:
        try:
            import requests as _req
        except ImportError:
            raise APIError("The 'requests' package is required. pip install requests")

        payload: Dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
            **kwargs,
        }

        try:
            r = _req.post(
                _ENDPOINT,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except _req.RequestException as exc:
            raise APIError(str(exc)) from exc

        if r.status_code == 401:
            raise AuthenticationError(
                detail=f"HTTP 401 from {_ENDPOINT}. "
                       f"Verify your key at {_SIGNUP_URL}"
            )
        if r.status_code == 403:
            raise AuthenticationError(
                detail=f"HTTP 403 — access denied. "
                       f"Check your key at {_SIGNUP_URL}"
            )
        if r.status_code == 429:
            retry_after = None
            try:
                retry_after = float(r.headers.get("Retry-After", 0)) or None
            except (TypeError, ValueError):
                pass
            raise RateLimitError(retry_after=retry_after)
        if not r.ok:
            raise APIError(
                f"API error {r.status_code}: {r.text[:300]}",
                code=r.status_code,
            )

        try:
            data = r.json()
        except ValueError as exc:
            raise APIError(f"Failed to parse JSON response: {exc}") from exc

        try:
            text       = data["choices"][0]["message"]["content"]
            usage      = data.get("usage", {})
            used_model = data.get("model", model)
        except (KeyError, IndexError):
            text       = str(data)
            usage      = {}
            used_model = model

        return AIResponse(text=text or "", model=used_model, usage=usage, raw=data)

    def _stream(
        self,
        messages: List[Dict],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Generator[str, None, None]:
        try:
            import requests as _req
        except ImportError:
            raise APIError("The 'requests' package is required. pip install requests")

        payload: Dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      True,
            **kwargs,
        }

        try:
            r = _req.post(
                _ENDPOINT,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
        except _req.RequestException as exc:
            raise APIError(str(exc)) from exc

        if r.status_code == 401:
            raise AuthenticationError(
                detail=f"HTTP 401. Verify your key at {_SIGNUP_URL}"
            )
        if r.status_code == 403:
            raise AuthenticationError(
                detail=f"HTTP 403. Check your key at {_SIGNUP_URL}"
            )
        if r.status_code == 429:
            raise RateLimitError()
        if not r.ok:
            raise APIError(f"API error {r.status_code}: {r.text[:300]}", code=r.status_code)

        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk   = json.loads(data_str)
                content = chunk["choices"][0].get("delta", {}).get("content")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    def __repr__(self) -> str:
        return (
            f"AIClient(endpoint={_ENDPOINT!r}, model={self.model!r}, "
            f"key={'***' if self.api_key else 'None'})"
        )


# =============================================================================
# WizardAI Core — top-level orchestrator
# =============================================================================

class WizardAI:
    """All-in-one WizardAI orchestrator.

    Combines:
    - :class:`AIClient`            — Sagittarius Labs AI
    - :class:`ConversationAgent`   — pattern-matched chat
    - :class:`MemoryManager`       — conversation memory
    - :class:`VisionModule`        — camera & computer vision
    - :class:`SpeechModule`        — STT & TTS
    - :class:`PluginManager`       — extensible skills
    - :class:`FileHelper`          — file utilities
    - :class:`DataSerializer`      — data persistence

    Example::

        import wizardai

        wiz = wizardai.WizardAI(api_key="YOUR_KEY")
        wiz.start()

        # Pattern-matched response (no API call)
        wiz.agent.add_pattern("hello", "Hello from WizardAI!")
        print(wiz.chat("hello"))

        # Direct LLM call
        print(wiz.ask("What is the capital of France?"))

        # Streaming
        for chunk in wiz.ai.chat_stream([{"role": "user", "content": "Write a poem"}]):
            print(chunk, end="", flush=True)

        wiz.stop()

    Get your API key at: https://sagittarius-labs.pages.dev/
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        # AI
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        # Vision
        enable_vision: bool = False,
        camera_device: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        # Speech
        enable_speech: bool = False,
        stt_backend: str = "google",
        tts_backend: str = "pyttsx3",
        language: str = "en-US",
        # Conversation agent
        agent_name: str = "WizardBot",
        fallback_response: str = "I'm not sure how to respond to that.",
        # Memory
        max_history: int = 50,
        memory_path: Optional[str] = None,
        # System prompt
        system_prompt: Optional[str] = None,
        # Logging
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        # Storage
        data_dir: str = "./wizardai_data",
        **kwargs,
    ):
        """
        Args:
            api_key           : Sagittarius Labs API key.
                                Falls back to ``WIZARDAI_API_KEY`` env var.
                                Obtain one at https://sagittarius-labs.pages.dev/
            model             : Override the default model.
            max_tokens        : Default max tokens for LLM responses.
            temperature       : Default sampling temperature.
            enable_vision     : Open the webcam on :meth:`start`.
            camera_device     : OpenCV camera index.
            camera_width      : Capture width in pixels.
            camera_height     : Capture height in pixels.
            enable_speech     : Initialise STT/TTS on :meth:`start`.
            stt_backend       : 'google' | 'sphinx' | 'whisper'.
            tts_backend       : 'pyttsx3' | 'gtts' | 'elevenlabs'.
            language          : BCP-47 language code (e.g. 'en-US').
            agent_name        : Display name of the conversation agent.
            fallback_response : Response when no pattern matches.
            max_history       : Sliding window size for conversation memory.
            memory_path       : Path for persistent memory (JSON).
            system_prompt     : Default LLM system prompt.
            log_level         : 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'.
            log_file          : Optional path to write logs to disk.
            data_dir          : Working directory for data persistence.
        """
        # ------------------------------------------------------------------
        self.logger = Logger("WizardAI", level=log_level, log_file=log_file)
        self.logger.info(f"WizardAI v{self.VERSION} initialising…")

        # ------------------------------------------------------------------
        self.data_dir   = Path(data_dir)
        self.files      = FileHelper(base_dir=self.data_dir)
        self.serializer = DataSerializer()

        # ------------------------------------------------------------------
        self.memory = MemoryManager(
            max_history=max_history,
            persist_path=memory_path,
            logger=self.logger,
        )

        # ------------------------------------------------------------------
        self.ai = AIClient(
            api_key=api_key,
            model=model or _MODEL,
            logger=self.logger,
        )

        self._max_tokens   = max_tokens
        self._temperature  = temperature
        self._system_prompt = system_prompt

        # ------------------------------------------------------------------
        self.agent = ConversationAgent(
            name=agent_name,
            fallback=fallback_response,
            memory=self.memory,
            logger=self.logger,
        )

        # ------------------------------------------------------------------
        self.plugins = PluginManager(logger=self.logger)

        # ------------------------------------------------------------------
        self._enable_vision = enable_vision
        self._camera_device = camera_device
        self._camera_width  = camera_width
        self._camera_height = camera_height
        self.vision: Optional[VisionModule] = None

        # ------------------------------------------------------------------
        self._enable_speech = enable_speech
        self._stt_backend   = stt_backend
        self._tts_backend   = tts_backend
        self._language      = language
        self.speech: Optional[SpeechModule] = None

        # ------------------------------------------------------------------
        self._running = False
        self.logger.info("WizardAI initialised successfully.")
        self.logger.info(f"  API key : {'set' if self.ai.api_key else f'NOT SET — visit {_SIGNUP_URL}'}")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the WizardAI session.

        Opens the camera (if ``enable_vision=True``),
        initialises speech (if ``enable_speech=True``),
        and calls ``on_start()`` on all registered plugins.
        """
        if self._running:
            self.logger.warning("WizardAI is already running.")
            return
        if self._enable_vision:
            self._init_vision()
        if self._enable_speech:
            self._init_speech()
        self.plugins.start_all()
        self._running = True
        self.logger.info("WizardAI session started.")

    def stop(self):
        """Stop the session and release all resources."""
        if not self._running:
            return
        self.plugins.stop_all()
        if self.speech:
            self.speech.stop_continuous_listening()
        if self.vision:
            self.vision.close()
        self.memory.save()
        self._running = False
        self.logger.info("WizardAI session stopped.")

    # ------------------------------------------------------------------
    def _init_vision(self):
        try:
            self.vision = VisionModule(
                device_id=self._camera_device,
                width=self._camera_width,
                height=self._camera_height,
                logger=self.logger,
            )
            self.vision.open()
        except Exception as exc:
            self.logger.error(f"Vision init failed: {exc}")
            self.vision = None

    def _init_speech(self):
        try:
            self.speech = SpeechModule(
                stt_backend=self._stt_backend,
                tts_backend=self._tts_backend,
                language=self._language,
                logger=self.logger,
            )
        except Exception as exc:
            self.logger.error(f"Speech init failed: {exc}")
            self.speech = None

    # ------------------------------------------------------------------
    # Chat interface
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """Process user input through the full pipeline.

        Priority:
        1. Plugin dispatch.
        2. Pattern-matched rules.
        3. LLM fallback via :meth:`ask`.

        Returns:
            Response string.
        """
        plugin_resp = self.plugins.dispatch(user_input, context={})
        if plugin_resp is not None:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", plugin_resp)
            return plugin_resp

        response = self.agent.respond(user_input)

        if response == self.agent.fallback and self.ai.api_key:
            try:
                llm_reply = self.ask(user_input)
                msgs = self.memory.get_history()
                if msgs and msgs[-1].role == "assistant":
                    msgs[-1].content = llm_reply
                return llm_reply
            except Exception as exc:
                self.logger.warning(f"LLM fallback failed: {exc}")

        return response

    def ask(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
        image_b64: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Send a prompt directly to the AI and return the response text.

        Unlike :meth:`chat`, this bypasses pattern matching and goes
        straight to the Sagittarius Labs LLM.

        Args:
            prompt          : User prompt string.
            model           : Override the default model.
            max_tokens      : Override max tokens.
            temperature     : Override temperature.
            system_prompt   : Override the default system prompt.
            include_history : If True, full conversation history is sent.
            image_b64       : Base64-encoded image for multimodal requests.

        Returns:
            Generated text string.
        """
        messages = self.memory.get_messages_for_api() if include_history else []

        user_content: Any = prompt
        if image_b64:
            user_content = [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        messages.append({"role": "user", "content": user_content})

        response: AIResponse = self.ai.chat(
            messages=messages,
            model=model,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            system_prompt=system_prompt or self._system_prompt,
            **kwargs,
        )

        self.memory.add_message("user", prompt)
        self.memory.add_message("assistant", response.text)
        return response.text

    def ask_raw(self, prompt: str, **kwargs) -> AIResponse:
        """Like :meth:`ask` but returns the full :class:`AIResponse` object."""
        return self.ai.complete(prompt, **kwargs)

    # ------------------------------------------------------------------
    # Speech shortcuts
    # ------------------------------------------------------------------

    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """Capture and transcribe speech from the microphone."""
        if not self.speech:
            self.logger.warning("Speech module not enabled. Pass enable_speech=True.")
            return None
        try:
            return self.speech.listen(timeout=timeout)
        except Exception as exc:
            self.logger.error(f"listen() failed: {exc}")
            return None

    def say(self, text: str, blocking: bool = True):
        """Speak *text* aloud using the TTS engine."""
        if not self.speech:
            self.logger.warning("Speech module not enabled. Pass enable_speech=True.")
            return
        try:
            self.speech.say(text, blocking=blocking)
        except Exception as exc:
            self.logger.error(f"say() failed: {exc}")

    def voice_chat(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for speech, respond, and speak the response back."""
        text = self.listen(timeout=timeout)
        if not text:
            return None
        self.logger.info(f"[VoiceChat] You said: {text!r}")
        response = self.chat(text)
        self.logger.info(f"[VoiceChat] Bot: {response!r}")
        self.say(response)
        return response

    # ------------------------------------------------------------------
    # Vision shortcuts
    # ------------------------------------------------------------------

    def capture(self) -> Optional[Any]:
        """Capture and return a single camera frame (numpy ndarray)."""
        if not self.vision:
            self.logger.warning("Vision module not enabled. Pass enable_vision=True.")
            return None
        return self.vision.capture_frame()

    def snapshot(self, path: Union[str, Path] = "snapshot.jpg") -> Optional[Path]:
        """Capture a frame and save it to *path*."""
        frame = self.capture()
        if frame is None:
            return None
        return self.vision.save_frame(frame, path)

    # ------------------------------------------------------------------
    # Memory shortcuts
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any):
        """Store a fact in long-term memory."""
        self.memory.remember(key, value)

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact from long-term memory."""
        return self.memory.recall(key, default)

    def get_history(self, n: int = 10) -> List[Dict]:
        """Return the last *n* conversation turns."""
        return self.memory.get_history_as_dicts(n)

    # ------------------------------------------------------------------
    # Plugin shortcuts
    # ------------------------------------------------------------------

    def add_plugin(
        self,
        plugin_cls: Type[PluginBase],
        config: Optional[Dict[str, Any]] = None,
    ) -> PluginBase:
        """Register a plugin class."""
        return self.plugins.register(plugin_cls, config=config)

    def load_plugins_from_dir(self, directory: Union[str, Path]) -> List[PluginBase]:
        """Load all plugins from a directory of Python files."""
        return self.plugins.load_from_directory(directory)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_system_prompt(self, prompt: str):
        """Update the default system prompt for all LLM calls."""
        self._system_prompt = prompt

    def set_model(self, model: str):
        """Switch the default model."""
        self.ai.set_model(model)

    def set_api_key(self, api_key: str):
        """Update the API key at runtime."""
        self.ai.set_api_key(api_key)

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------

    def run_repl(
        self,
        prompt_str: str = "You: ",
        quit_commands: Optional[List[str]] = None,
        voice_mode: bool = False,
    ):
        """Start an interactive chat REPL in the terminal.

        Args:
            prompt_str    : Input prompt shown to the user.
            quit_commands : Commands that exit the loop.
            voice_mode    : If True, use voice I/O.
        """
        quit_cmds = set(quit_commands or ["quit", "exit", "bye", "/q"])
        print(f"\n{'='*55}")
        print(f"  WizardAI v{self.VERSION}  –  {self.agent.name}")
        print(f"  Type one of {quit_cmds} to exit.")
        print(f"{'='*55}\n")

        def _signal_handler(sig, frame):
            print("\nInterrupted. Stopping WizardAI…")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)

        while True:
            try:
                if voice_mode and self.speech:
                    print("[Listening…]")
                    user_input = self.listen()
                    if not user_input:
                        continue
                    print(f"You said: {user_input}")
                else:
                    user_input = input(prompt_str).strip()

                if not user_input:
                    continue
                if user_input.lower() in quit_cmds:
                    print("Goodbye!")
                    break

                response = self.chat(user_input)
                print(f"{self.agent.name}: {response}\n")

                if voice_mode and self.speech:
                    self.say(response)

            except EOFError:
                break
            except Exception as exc:
                self.logger.error(f"REPL error: {exc}")

        self.stop()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return (
            f"WizardAI(version={self.VERSION!r}, "
            f"model={self.ai.model!r}, status={status})"
        )


# =============================================================================
# Public API surface
# =============================================================================

__all__ = [
    # Core
    "WizardAI",
    # AI
    "AIClient",
    "AIResponse",
    # Conversation
    "ConversationAgent",
    "Pattern",
    # Memory
    "MemoryManager",
    "Message",
    # Vision
    "VisionModule",
    # Speech
    "SpeechModule",
    # Plugins
    "PluginBase",
    "PluginManager",
    # Utils
    "Logger",
    "FileHelper",
    "DataSerializer",
    "RateLimiter",
    # Exceptions
    "WizardAIError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "VisionError",
    "CameraNotFoundError",
    "SpeechError",
    "MicrophoneNotFoundError",
    "ConversationError",
    "PluginError",
    "ConfigurationError",
    # Constants
    "__version__",
    "__author__",
    "__license__",
]
