"""
WizardAI Core
-------------
Top-level orchestrator that wires together the AI client, vision module,
speech module, conversation agent, memory manager, and plugin system into
a single, easy-to-use interface.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .ai_client import AIBackend, AIClient, AIResponse
from .conversation import ConversationAgent
from .exceptions import WizardAIError
from .memory import MemoryManager
from .plugins import PluginBase, PluginManager
from .utils import DataSerializer, FileHelper, Logger


# ---------------------------------------------------------------------------
# WizardAI
# ---------------------------------------------------------------------------

class WizardAI:
    """All-in-one WizardAI orchestrator.

    Provides a single entry point that integrates:
    - :class:`~wizardai.ai_client.AIClient`  – multi-backend LLM calls
    - :class:`~wizardai.vision.VisionModule` – camera & computer vision
    - :class:`~wizardai.speech.SpeechModule` – STT & TTS
    - :class:`~wizardai.conversation.ConversationAgent` – AIML-style chat
    - :class:`~wizardai.memory.MemoryManager` – conversation memory
    - :class:`~wizardai.plugins.PluginManager` – extensible skills

    Example (minimal)::

        import wizardai

        wiz = wizardai.WizardAI(openai_api_key="sk-...")
        wiz.start()

        # Pattern-matched chat
        wiz.agent.add_pattern("hello", "Hello from WizardAI!")
        reply = wiz.chat("hello")
        print(reply)

        # LLM call
        reply = wiz.ask("What is the capital of France?")
        print(reply)

        wiz.stop()

    Example (full features)::

        wiz = wizardai.WizardAI(
            openai_api_key="sk-...",
            enable_vision=True,
            enable_speech=True,
            stt_backend="google",
            tts_backend="pyttsx3",
        )
        wiz.start()

        # Capture + describe a frame
        frame   = wiz.vision.capture_frame()
        b64     = wiz.vision.encode_to_base64(frame)
        caption = wiz.ask("Describe this image briefly.", image_b64=b64)
        wiz.speech.say(caption)

        wiz.stop()
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        # AI backend
        ai_backend: Union[str, AIBackend] = AIBackend.OPENAI,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        custom_endpoint: Optional[str] = None,
        default_model: Optional[str] = None,
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
        # Agent
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
            ai_backend:        Backend for LLM calls ('openai', 'anthropic', …).
            openai_api_key:    OpenAI API key (falls back to OPENAI_API_KEY env).
            anthropic_api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY).
            huggingface_api_key: HF API key (falls back to HUGGINGFACE_API_KEY).
            custom_endpoint:   Base URL for a custom OpenAI-compatible endpoint.
            default_model:     Override the default model for the chosen backend.
            max_tokens:        Default max tokens for LLM calls.
            temperature:       Default temperature for LLM calls.
            enable_vision:     If True, open the camera on :meth:`start`.
            camera_device:     OpenCV camera device index.
            camera_width:      Capture width in pixels.
            camera_height:     Capture height in pixels.
            enable_speech:     If True, initialise STT/TTS on :meth:`start`.
            stt_backend:       Speech recognition backend.
            tts_backend:       TTS engine.
            language:          BCP-47 language code.
            agent_name:        Display name of the conversation agent.
            fallback_response: Agent response when no pattern matches.
            max_history:       Number of messages kept in memory.
            memory_path:       Path for persistent memory storage.
            system_prompt:     Default system prompt for LLM calls.
            log_level:         Logging level (DEBUG|INFO|WARNING|ERROR).
            log_file:          Optional log file path.
            data_dir:          Directory for data persistence.
            **kwargs:          Additional keyword args forwarded to AIClient.
        """
        # ------------------------------------------------------------------
        # Logger
        # ------------------------------------------------------------------
        self.logger = Logger("WizardAI", level=log_level, log_file=log_file)
        self.logger.info(f"WizardAI v{self.VERSION} initialising…")

        # ------------------------------------------------------------------
        # Storage helpers
        # ------------------------------------------------------------------
        self.data_dir = Path(data_dir)
        self.files = FileHelper(base_dir=self.data_dir)
        self.serializer = DataSerializer()

        # ------------------------------------------------------------------
        # Memory
        # ------------------------------------------------------------------
        self.memory = MemoryManager(
            max_history=max_history,
            persist_path=memory_path,
            logger=self.logger,
        )

        # ------------------------------------------------------------------
        # AI Client
        # ------------------------------------------------------------------
        backend = AIBackend(ai_backend) if isinstance(ai_backend, str) else ai_backend

        # Resolve API key based on backend
        _api_key = (
            openai_api_key
            or anthropic_api_key
            or huggingface_api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("HUGGINGFACE_API_KEY", "")
        )
        if backend == AIBackend.OPENAI and openai_api_key:
            _api_key = openai_api_key
        elif backend == AIBackend.ANTHROPIC and anthropic_api_key:
            _api_key = anthropic_api_key
        elif backend == AIBackend.HUGGINGFACE and huggingface_api_key:
            _api_key = huggingface_api_key

        self.ai = AIClient(
            backend=backend,
            api_key=_api_key,
            model=default_model,
            endpoint=custom_endpoint,
            logger=self.logger,
            **kwargs,
        )

        # Defaults for LLM calls
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt

        # ------------------------------------------------------------------
        # Conversation agent
        # ------------------------------------------------------------------
        self.agent = ConversationAgent(
            name=agent_name,
            fallback=fallback_response,
            memory=self.memory,
            logger=self.logger,
        )

        # ------------------------------------------------------------------
        # Plugin manager
        # ------------------------------------------------------------------
        self.plugins = PluginManager(logger=self.logger)

        # ------------------------------------------------------------------
        # Vision (lazy)
        # ------------------------------------------------------------------
        self._enable_vision = enable_vision
        self._camera_device = camera_device
        self._camera_width = camera_width
        self._camera_height = camera_height
        self.vision = None  # initialised in start()

        # ------------------------------------------------------------------
        # Speech (lazy)
        # ------------------------------------------------------------------
        self._enable_speech = enable_speech
        self._stt_backend = stt_backend
        self._tts_backend = tts_backend
        self._language = language
        self.speech = None  # initialised in start()

        # ------------------------------------------------------------------
        # Session state
        # ------------------------------------------------------------------
        self._running = False
        self._session_callbacks: List[Callable] = []

        self.logger.info("WizardAI initialised successfully.")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the WizardAI session.

        - Opens the camera (if ``enable_vision=True``).
        - Initialises speech modules (if ``enable_speech=True``).
        - Calls ``on_start()`` on all registered plugins.
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
        """Stop the WizardAI session and release all resources."""
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
    # Lazy module initialisation
    # ------------------------------------------------------------------

    def _init_vision(self):
        """Lazily import and initialise VisionModule."""
        try:
            from .vision import VisionModule
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
        """Lazily import and initialise SpeechModule."""
        try:
            from .speech import SpeechModule
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
        """Process *user_input* through the conversation pipeline.

        Priority:
        1. Plugin dispatch (``!plugin_name args`` syntax).
        2. Pattern-matched rules in the :attr:`agent`.
        3. LLM fallback via :meth:`ask` (if AI backend is configured).

        Args:
            user_input: Raw user text.

        Returns:
            Response string.
        """
        # Plugin dispatch
        plugin_response = self.plugins.dispatch(user_input, context={})
        if plugin_response is not None:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", plugin_response)
            return plugin_response

        # Pattern-matched agent
        response = self.agent.respond(user_input)

        # If the agent fell back to its default and we have an LLM, use it
        if response == self.agent.fallback and self.ai.api_key:
            try:
                llm_reply = self.ask(user_input)
                # Update memory (agent already added turns, overwrite last)
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
        """Send a prompt directly to the AI backend and return the response text.

        Unlike :meth:`chat`, this bypasses the pattern-matching agent and
        sends the message straight to the LLM.

        Args:
            prompt:          User prompt string.
            model:           Override the default model.
            max_tokens:      Override max tokens.
            temperature:     Override temperature.
            system_prompt:   Override default system prompt.
            include_history: If True, the full conversation history is sent.
            image_b64:       Base64-encoded image for multimodal models.
            **kwargs:        Extra kwargs forwarded to :meth:`AIClient.chat`.

        Returns:
            Generated text string.
        """
        # Build message list
        if include_history:
            messages = self.memory.get_messages_for_api()
        else:
            messages = []

        # Append current user message
        user_content: Any = prompt
        if image_b64:
            # Vision-capable payload (OpenAI format)
            user_content = [
                {"type": "text", "text": prompt},
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

        # Update memory
        self.memory.add_message("user", prompt)
        self.memory.add_message("assistant", response.text)

        return response.text

    def ask_raw(self, prompt: str, **kwargs) -> AIResponse:
        """Like :meth:`ask` but returns the full :class:`AIResponse` object."""
        response = self.ai.complete(prompt, **kwargs)
        return response

    # ------------------------------------------------------------------
    # Speech shortcuts
    # ------------------------------------------------------------------

    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """Capture and transcribe speech from the microphone.

        Returns:
            Transcribed text, or None if speech is unavailable / fails.
        """
        if not self.speech:
            self.logger.warning("Speech module not enabled. Pass enable_speech=True.")
            return None
        try:
            return self.speech.listen(timeout=timeout)
        except Exception as exc:
            self.logger.error(f"listen() failed: {exc}")
            return None

    def say(self, text: str, blocking: bool = True):
        """Speak *text* aloud using the TTS engine.

        Args:
            text:     Text to speak.
            blocking: Wait for speech to finish.
        """
        if not self.speech:
            self.logger.warning("Speech module not enabled. Pass enable_speech=True.")
            return
        try:
            self.speech.say(text, blocking=blocking)
        except Exception as exc:
            self.logger.error(f"say() failed: {exc}")

    def voice_chat(self, timeout: float = 5.0) -> Optional[str]:
        """Listen for speech, chat, and speak the response.

        Returns:
            The agent response text, or None on error.
        """
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
        """Capture and return a single camera frame.

        Returns:
            Frame (numpy ndarray) or None if vision is unavailable.
        """
        if not self.vision:
            self.logger.warning("Vision module not enabled. Pass enable_vision=True.")
            return None
        return self.vision.capture_frame()

    def snapshot(self, path: Union[str, Path] = "snapshot.jpg") -> Optional[Path]:
        """Capture a frame and save it to *path*.

        Returns:
            The saved Path, or None on failure.
        """
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
        plugin_cls,
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
        """Update the default system prompt sent on every LLM call."""
        self._system_prompt = prompt

    def set_model(self, model: str):
        """Switch the default model."""
        self.ai.set_model(model)

    def set_api_key(self, api_key: str, backend: Optional[str] = None):
        """Update the API key (optionally switch backends)."""
        if backend:
            self.ai.backend = AIBackend(backend)
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
            prompt_str:     Input prompt shown to the user.
            quit_commands:  Commands that exit the loop (default: quit, exit, bye).
            voice_mode:     If True, use voice I/O instead of text.
        """
        quit_cmds = set(quit_commands or ["quit", "exit", "bye", "/q"])
        print(f"\n{'='*50}")
        print(f"  WizardAI v{self.VERSION}  –  {self.agent.name}")
        print(f"  Type one of {quit_cmds} to exit.")
        print(f"{'='*50}\n")

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

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        status = "running" if self._running else "stopped"
        return (
            f"WizardAI(version={self.VERSION!r}, backend={self.ai.backend.value!r}, "
            f"model={self.ai.model!r}, status={status})"
        )
