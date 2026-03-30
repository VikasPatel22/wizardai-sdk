"""
WizardAI Conversation Module
-----------------------------
AIML-style pattern-matching conversational agent with context/memory
management, wildcard support, priority rules, and a plugin system.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .exceptions import ConversationError
from .memory import MemoryManager
from .utils import Logger


# ---------------------------------------------------------------------------
# Pattern dataclass
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    """A single conversation rule mapping an input pattern to a response.

    Attributes:
        pattern:   Input pattern string. Supports wildcards:
                   ``*``  – matches any sequence of words.
                   ``?``  – matches exactly one word.
                   ``{n}`` – named capture group (accessible in template).
        template:  Response string, callable, or list of alternatives.
                   In string templates, ``{wildcard}`` inserts the matched
                   text, and ``{0}``, ``{1}``, … index capture groups.
        priority:  Higher priority patterns match first (default 0).
        context:   Optional context key this rule requires to be active.
        tags:      Arbitrary labels for grouping / filtering rules.
    """
    pattern: str
    template: Union[str, Callable[..., str], List[str]]
    priority: int = 0
    context: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Compiled regex – populated lazily by ConversationAgent
    _regex: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self):
        self._regex = None  # force lazy compile

    def compile(self) -> re.Pattern:
        """Compile the pattern to a regular expression."""
        if self._regex is None:
            self._regex = _pattern_to_regex(self.pattern)
        return self._regex


# ---------------------------------------------------------------------------
# Pattern compilation helpers
# ---------------------------------------------------------------------------

def _pattern_to_regex(pattern: str) -> re.Pattern:
    """Convert a WizardAI pattern string to a compiled regex.

    Syntax:
    - ``*``  → matches one or more words (greedy).
    - ``?``  → matches exactly one word.
    - ``{name}`` → named capture group matching one or more words.
    - All other characters are treated as literals.
    """
    # Escape special regex chars except our wildcards
    tokens = re.split(r"(\*|\?|\{[^}]+\})", pattern)
    parts = []
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


# ---------------------------------------------------------------------------
# ConversationAgent
# ---------------------------------------------------------------------------

class ConversationAgent:
    """AIML-style conversational agent with wildcards, context, and plugins.

    The agent matches user input against registered :class:`Pattern` rules in
    priority order.  When a match is found, the response template is rendered
    (with wildcard substitution) and returned.  If no rule matches, the
    fallback response is used.

    The agent integrates with :class:`~wizardai.memory.MemoryManager` for
    conversation history and supports a plugin system for custom skills.

    Example::

        agent = ConversationAgent(name="WizardBot")

        # Register rules
        agent.add_pattern("hello", "Hello! How can I help you?")
        agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")
        agent.add_pattern(
            "what time is it",
            lambda: f"It's {time.strftime('%H:%M')}."
        )

        # Chat loop
        while True:
            user_input = input("You: ")
            reply = agent.respond(user_input)
            print(f"Bot: {reply}")
    """

    def __init__(
        self,
        name: str = "WizardBot",
        fallback: str = "I'm not sure how to respond to that.",
        memory: Optional[MemoryManager] = None,
        logger: Optional[Logger] = None,
        case_sensitive: bool = False,
    ):
        """
        Args:
            name:           Display name of the agent.
            fallback:       Response used when no pattern matches.
            memory:         Optional :class:`MemoryManager` for history.
            logger:         Optional Logger instance.
            case_sensitive: If True, pattern matching is case-sensitive.
        """
        self.name = name
        self.fallback = fallback
        self.memory = memory or MemoryManager(max_history=100)
        self.logger = logger or Logger("ConversationAgent")
        self.case_sensitive = case_sensitive

        self._patterns: List[Pattern] = []
        self._active_context: Optional[str] = None
        self._plugins: Dict[str, Callable] = {}
        self._preprocessors: List[Callable[[str], str]] = []
        self._postprocessors: List[Callable[[str], str]] = []

        # Built-in default rules
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
        """Register a new conversation rule.

        Args:
            pattern:  Input pattern (supports ``*``, ``?``, ``{name}``).
            template: Response string, callable returning str, or list of
                      alternative strings (one is chosen at random).
            priority: Higher values match first.
            context:  If set, rule only activates when this context is active.
            tags:     Optional labels for grouping.

        Returns:
            The created :class:`Pattern` object.
        """
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
        self.logger.debug(f"[Agent] Pattern added: {pattern!r} (priority={priority})")
        return p

    def remove_pattern(self, pattern_str: str) -> int:
        """Remove all rules matching *pattern_str*.

        Returns:
            Number of rules removed.
        """
        before = len(self._patterns)
        self._patterns = [p for p in self._patterns if p.pattern != pattern_str]
        removed = before - len(self._patterns)
        self.logger.debug(f"[Agent] Removed {removed} rule(s) for pattern {pattern_str!r}")
        return removed

    def load_patterns_from_dict(self, rules: Dict[str, Any]):
        """Bulk-load rules from a dictionary.

        Expected format::

            {
                "hello": "Hello there!",
                "bye|goodbye": ["See you!", "Goodbye!", "Take care!"],
                "what is *": "I don't know what {wildcard} is yet."
            }
        """
        for pattern, template in rules.items():
            self.add_pattern(pattern, template)
        self.logger.info(f"[Agent] Loaded {len(rules)} pattern(s) from dict.")

    def load_patterns_from_file(self, path: str):
        """Load rules from a JSON file.

        The JSON file should contain a dict mapping patterns to templates.
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        self.load_patterns_from_dict(rules)

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def respond(self, user_input: str) -> str:
        """Generate a response to *user_input*.

        The pipeline:
        1. Pre-process input.
        2. Match against registered patterns (highest priority first).
        3. Render the matched template (wildcard substitution).
        4. Post-process the response.
        5. Store both turns in memory.

        Args:
            user_input: Raw text from the user.

        Returns:
            Agent response string.
        """
        # Pre-process
        processed = self._preprocess(user_input)

        # Match
        response, matched_pattern = self._match(processed)

        # Post-process
        response = self._postprocess(response)

        # Memory
        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", response)

        self.logger.debug(
            f"[Agent] Input={processed!r}, "
            f"Pattern={matched_pattern!r}, "
            f"Response={response[:60]!r}"
        )
        return response

    def _match(self, text: str) -> Tuple[str, Optional[str]]:
        """Find the highest-priority pattern that matches *text*.

        Returns:
            (response_text, matched_pattern_str)
        """
        for pattern in self._patterns:
            # Context check
            if pattern.context and pattern.context != self._active_context:
                continue

            regex = pattern.compile()
            m = regex.match(text)
            if m:
                response = self._render_template(pattern.template, m)
                # Update context if pattern defines one
                if pattern.context:
                    self._active_context = pattern.context
                return response, pattern.pattern

        return self.fallback, None

    def _render_template(
        self,
        template: Union[str, Callable, List[str]],
        match: re.Match,
    ) -> str:
        """Render a response template using match groups."""
        import random

        # Resolve callable
        if callable(template):
            try:
                result = template()
                return str(result)
            except Exception as exc:
                self.logger.error(f"Template callable error: {exc}")
                return self.fallback

        # Pick from list
        if isinstance(template, list):
            template = random.choice(template)

        # Substitute wildcards
        groups = match.groups()
        named = match.groupdict()

        # {wildcard} → first anonymous capture group
        if "{wildcard}" in template and groups:
            template = template.replace("{wildcard}", groups[0])

        # {0}, {1}, … → positional capture groups
        for i, group in enumerate(groups):
            template = template.replace(f"{{{i}}}", group or "")

        # {name} → named capture group
        for name, value in named.items():
            template = template.replace(f"{{{name}}}", value or "")

        # {memory:key} → long-term memory lookup
        template = re.sub(
            r"\{memory:([^}]+)\}",
            lambda m: str(self.memory.recall(m.group(1), "")),
            template,
        )

        return template

    # ------------------------------------------------------------------
    # Pre / post processors
    # ------------------------------------------------------------------

    def add_preprocessor(self, fn: Callable[[str], str]):
        """Register a function to transform user input before matching.

        Preprocessors run in registration order.

        Example::

            agent.add_preprocessor(str.lower)
            agent.add_preprocessor(lambda t: t.strip("!?."))
        """
        self._preprocessors.append(fn)

    def add_postprocessor(self, fn: Callable[[str], str]):
        """Register a function to transform agent output after matching."""
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
    # Plugins
    # ------------------------------------------------------------------

    def register_plugin(self, name: str, handler: Callable):
        """Register a named plugin handler.

        Plugins are callable skills invokable via the ``!plugin_name``
        prefix in user input.

        Example::

            agent.register_plugin("weather", get_weather)
            # User types: "!weather London"
            # Calls:      get_weather("London")
        """
        self._plugins[name.lower()] = handler
        self.logger.info(f"[Agent] Plugin registered: {name!r}")

    def _dispatch_plugin(self, text: str) -> Optional[str]:
        """Check if *text* is a plugin invocation and dispatch if so."""
        if not text.startswith("!"):
            return None
        parts = text[1:].split(None, 1)
        if not parts:
            return None
        name = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
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
        """Manually activate a named context."""
        self._active_context = context
        self.logger.debug(f"[Agent] Context set to: {context!r}")

    def clear_context(self):
        """Clear the active context."""
        self._active_context = None

    # ------------------------------------------------------------------
    # Built-in defaults
    # ------------------------------------------------------------------

    def _register_defaults(self):
        """Add sensible built-in rules."""
        defaults = {
            "hello": ["Hello!", "Hi there!", "Hey! How can I help you?"],
            "hi": ["Hi!", "Hello!", "Hey!"],
            "how are you": [
                "I'm doing well, thanks for asking!",
                "Great! Ready to assist you.",
            ],
            "what is your name": f"I'm {self.name}, your AI assistant.",
            "what can you do": (
                "I can answer questions, hold conversations, and more. "
                "Just ask me anything!"
            ),
            "goodbye": ["Goodbye! Have a great day!", "See you later!", "Bye!"],
            "bye": ["Bye!", "Goodbye!", "Take care!"],
            "thank you": ["You're welcome!", "Happy to help!", "Anytime!"],
            "thanks": ["No problem!", "Glad I could help!", "Sure thing!"],
        }
        for pat, tmpl in defaults.items():
            self.add_pattern(pat, tmpl, priority=-10)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_patterns(self, tag: Optional[str] = None) -> List[Pattern]:
        """Return registered patterns, optionally filtered by *tag*."""
        if tag:
            return [p for p in self._patterns if tag in p.tags]
        return list(self._patterns)

    def get_history(self, n: int = 10) -> List[Dict]:
        """Return the last *n* conversation turns as a list of dicts."""
        return self.memory.get_history_as_dicts(n)

    def reset(self):
        """Clear conversation history and reset context."""
        self.memory.clear_history()
        self._active_context = None
        self.logger.info("[Agent] Conversation reset.")

    def __repr__(self):
        return (
            f"ConversationAgent(name={self.name!r}, "
            f"patterns={len(self._patterns)}, "
            f"context={self._active_context!r})"
        )
