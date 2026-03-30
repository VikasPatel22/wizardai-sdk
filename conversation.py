"""
WizardAI AI Client
------------------
Unified interface for multiple AI backends: OpenAI, Anthropic, Hugging Face,
and arbitrary custom REST endpoints.  Includes API-key management, model
selection, rate limiting, and automatic retry with exponential back-off.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union

from .exceptions import APIError, AuthenticationError, RateLimitError
from .utils import Logger, RateLimiter


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class AIBackend(str, Enum):
    """Supported AI backend identifiers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIResponse:
    """Structured response returned by :class:`AIClient`.

    Attributes:
        text:         The generated text content.
        model:        Model identifier used for generation.
        backend:      Backend that produced the response.
        usage:        Token / resource usage statistics.
        raw:          The raw response dict from the API.
        latency_ms:   Round-trip latency in milliseconds.
    """
    text: str
    model: str
    backend: AIBackend
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def __str__(self):
        return self.text


# ---------------------------------------------------------------------------
# AIClient
# ---------------------------------------------------------------------------

class AIClient:
    """Unified client for multiple AI backends.

    Handles API-key management (env vars or explicit args), model selection,
    rate limiting, and retry logic.  All backends expose the same
    :meth:`complete` / :meth:`chat` interface.

    Example::

        # OpenAI
        client = AIClient(backend="openai", api_key="sk-...")
        response = client.chat([{"role": "user", "content": "Hello!"}])
        print(response.text)

        # Anthropic
        client = AIClient(backend="anthropic", api_key="sk-ant-...")
        response = client.chat([{"role": "user", "content": "Hello!"}],
                               model="claude-3-5-sonnet-20241022")

        # Custom endpoint
        client = AIClient(backend="custom",
                          endpoint="https://my-api.example.com/v1/chat")
        response = client.chat([{"role": "user", "content": "Hello!"}])
    """

    # Default model names per backend
    _DEFAULT_MODELS: Dict[str, str] = {
        AIBackend.OPENAI: "gpt-4o-mini",
        AIBackend.ANTHROPIC: "claude-3-5-haiku-20241022",
        AIBackend.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2",
        AIBackend.CUSTOM: "default",
    }

    # Environment variable names to look up API keys
    _ENV_KEYS: Dict[str, str] = {
        AIBackend.OPENAI: "OPENAI_API_KEY",
        AIBackend.ANTHROPIC: "ANTHROPIC_API_KEY",
        AIBackend.HUGGINGFACE: "HUGGINGFACE_API_KEY",
        AIBackend.CUSTOM: "WIZARDAI_CUSTOM_API_KEY",
    }

    def __init__(
        self,
        backend: Union[str, AIBackend] = AIBackend.OPENAI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        rate_limit_calls: int = 60,
        rate_limit_period: float = 60.0,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        """
        Args:
            backend:            AI backend to use.
            api_key:            API key.  Falls back to environment variable
                                if not provided.
            model:              Model identifier.  Falls back to the backend
                                default if not provided.
            endpoint:           Base URL for custom / self-hosted endpoints.
            max_retries:        Number of retry attempts on transient errors.
            retry_delay:        Initial delay (seconds) between retries
                                (doubles on each attempt).
            timeout:            HTTP request timeout in seconds.
            rate_limit_calls:   Max API calls per *rate_limit_period* seconds.
            rate_limit_period:  Rate-limit window in seconds.
            logger:             Optional Logger instance.
            **kwargs:           Extra keyword args forwarded to the HTTP client.
        """
        self.backend = AIBackend(backend) if isinstance(backend, str) else backend
        self.model = model or self._DEFAULT_MODELS[self.backend]
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = logger or Logger("AIClient")
        self._extra = kwargs

        # Resolve API key
        self.api_key = api_key or os.environ.get(self._ENV_KEYS[self.backend], "")
        if not self.api_key and self.backend != AIBackend.CUSTOM:
            self.logger.warning(
                f"No API key found for backend '{self.backend.value}'. "
                f"Set env var {self._ENV_KEYS[self.backend]} or pass api_key=."
            )

        # Rate limiter
        self._rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)

        self.logger.info(
            f"AIClient initialised: backend={self.backend.value}, model={self.model}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> AIResponse:
        """Send a multi-turn chat request to the configured backend.

        Args:
            messages:      List of ``{"role": ..., "content": ...}`` dicts.
            model:         Override the default model for this call.
            max_tokens:    Maximum tokens to generate.
            temperature:   Sampling temperature (0 = deterministic).
            system_prompt: Prepend a system message to the conversation.
            stream:        If True, stream the response (returns generator).
            **kwargs:      Extra parameters forwarded to the backend.

        Returns:
            An :class:`AIResponse` object.

        Raises:
            APIError: On non-retryable API failures.
            RateLimitError: When the rate limit is exceeded.
            AuthenticationError: When the API key is invalid.
        """
        _model = model or self.model
        _messages = list(messages)

        if system_prompt:
            _messages = [{"role": "system", "content": system_prompt}] + _messages

        dispatch = {
            AIBackend.OPENAI: self._call_openai,
            AIBackend.ANTHROPIC: self._call_anthropic,
            AIBackend.HUGGINGFACE: self._call_huggingface,
            AIBackend.CUSTOM: self._call_custom,
        }
        call_fn = dispatch[self.backend]

        return self._with_retry(
            call_fn,
            messages=_messages,
            model=_model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AIResponse:
        """Single-turn text completion convenience wrapper.

        Wraps :meth:`chat` with a single user message.

        Args:
            prompt:     The input prompt string.
            model:      Override the default model.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            An :class:`AIResponse` object.
        """
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def set_model(self, model: str):
        """Change the default model used by this client."""
        self.model = model
        self.logger.info(f"Model changed to: {model}")

    def set_api_key(self, api_key: str):
        """Update the API key at runtime."""
        self.api_key = api_key
        self.logger.info("API key updated.")

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _with_retry(self, fn, **kwargs) -> AIResponse:
        """Call *fn* with retry / back-off on transient errors."""
        self._rate_limiter.wait()

        last_error = None
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 2):
            try:
                start = time.monotonic()
                response = fn(**kwargs)
                response.latency_ms = (time.monotonic() - start) * 1000
                return response
            except RateLimitError as exc:
                last_error = exc
                wait = exc.retry_after or delay
                self.logger.warning(
                    f"Rate limited. Waiting {wait:.1f}s before retry {attempt}…"
                )
                time.sleep(wait)
                delay *= 2
            except AuthenticationError:
                raise  # never retry auth failures
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
                last_error = APIError(str(exc), backend=self.backend.value)
                if attempt <= self.max_retries:
                    self.logger.warning(
                        f"Unexpected error (attempt {attempt}): {exc}. "
                        f"Retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise last_error from exc

        raise last_error or APIError("Max retries exceeded", backend=self.backend.value)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _call_openai(self, messages, model, max_tokens, temperature, **kwargs) -> AIResponse:
        """Call the OpenAI chat completions endpoint."""
        try:
            import openai
        except ImportError:
            raise APIError(
                "The 'openai' package is required for the OpenAI backend. "
                "Install it with: pip install openai",
                backend="openai",
            )

        client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        except openai.AuthenticationError as exc:
            raise AuthenticationError("openai") from exc
        except openai.RateLimitError as exc:
            raise RateLimitError() from exc
        except openai.APIError as exc:
            raise APIError(str(exc), backend="openai") from exc

        choice = resp.choices[0].message
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        return AIResponse(
            text=choice.content or "",
            model=resp.model,
            backend=self.backend,
            usage=usage,
            raw=resp.model_dump(),
        )

    def _call_anthropic(self, messages, model, max_tokens, temperature, **kwargs) -> AIResponse:
        """Call the Anthropic Messages API."""
        try:
            import anthropic
        except ImportError:
            raise APIError(
                "The 'anthropic' package is required. Install: pip install anthropic",
                backend="anthropic",
            )

        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

        # Anthropic separates system messages
        system_content = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                api_messages.append(m)

        call_kwargs: Dict[str, Any] = dict(
            model=model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        if system_content:
            call_kwargs["system"] = system_content

        try:
            resp = client.messages.create(**call_kwargs)
        except anthropic.AuthenticationError as exc:
            raise AuthenticationError("anthropic") from exc
        except anthropic.RateLimitError as exc:
            raise RateLimitError() from exc
        except anthropic.APIError as exc:
            raise APIError(str(exc), backend="anthropic") from exc

        text = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return AIResponse(
            text=text,
            model=resp.model,
            backend=self.backend,
            usage=usage,
            raw=resp.model_dump(),
        )

    def _call_huggingface(self, messages, model, max_tokens, temperature, **kwargs) -> AIResponse:
        """Call the Hugging Face Inference API."""
        try:
            import requests
        except ImportError:
            raise APIError("The 'requests' package is required.", backend="huggingface")

        base = self.endpoint or "https://api-inference.huggingface.co/models"
        url = f"{base}/{model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Format conversation as a single prompt string
        prompt = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )
        prompt += "\nAssistant:"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }

        import requests as _req
        try:
            r = _req.post(url, headers=headers, json=payload, timeout=self.timeout)
        except _req.RequestException as exc:
            raise APIError(str(exc), backend="huggingface") from exc

        if r.status_code == 401:
            raise AuthenticationError("huggingface")
        if r.status_code == 429:
            raise RateLimitError()
        if not r.ok:
            raise APIError(f"HuggingFace error {r.status_code}: {r.text}", backend="huggingface")

        data = r.json()
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "")
        else:
            text = str(data)

        return AIResponse(
            text=text,
            model=model,
            backend=self.backend,
            raw=data,
        )

    def _call_custom(self, messages, model, max_tokens, temperature, **kwargs) -> AIResponse:
        """Call a custom OpenAI-compatible REST endpoint."""
        if not self.endpoint:
            raise APIError(
                "A custom endpoint URL must be provided via the 'endpoint' parameter.",
                backend="custom",
            )

        try:
            import requests as _req
        except ImportError:
            raise APIError("The 'requests' package is required.", backend="custom")

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            r = _req.post(
                self.endpoint, headers=headers, json=payload, timeout=self.timeout
            )
        except _req.RequestException as exc:
            raise APIError(str(exc), backend="custom") from exc

        if r.status_code == 401:
            raise AuthenticationError("custom")
        if r.status_code == 429:
            raise RateLimitError()
        if not r.ok:
            raise APIError(
                f"Custom endpoint error {r.status_code}: {r.text}", backend="custom"
            )

        data = r.json()
        # Try to parse OpenAI-compatible format
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            text = str(data)

        return AIResponse(
            text=text,
            model=model,
            backend=self.backend,
            raw=data,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"AIClient(backend={self.backend.value!r}, model={self.model!r}, "
            f"key={'***' if self.api_key else 'None'})"
        )
