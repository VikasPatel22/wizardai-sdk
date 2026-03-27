# 🧙 WizardAI SDK

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/backends-OpenAI%20%7C%20Anthropic%20%7C%20HuggingFace-purple?style=for-the-badge" />
</p>

> **A powerful, all-in-one Python SDK for AI integration** — combining conversational AI, computer vision, speech I/O, memory management, and a flexible plugin system into a single, easy-to-use module.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [WizardAI Core](#wizardai-core)
  - [AIClient](#aiclient)
  - [ConversationAgent](#conversationagent)
  - [MemoryManager](#memorymanager)
  - [VisionModule](#visionmodule)
  - [SpeechModule](#speechmodule)
  - [Plugin System](#plugin-system)
  - [Exceptions](#exceptions)
- [Folder Structure](#folder-structure)
- [Configuration Reference](#configuration-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Description |
|---|---|
| 🤖 **Multi-backend LLM** | OpenAI (GPT-4o), Anthropic (Claude), HuggingFace, or any custom REST endpoint |
| 💬 **Conversation Agent** | AIML-style pattern matching with wildcards, priorities, and context |
| 🧠 **Memory Manager** | Short-term (sliding window) + long-term (key-value) + JSON persistence |
| 👁️ **Vision Module** | Real-time webcam capture, face detection, and frame streaming via OpenCV |
| 🎙️ **Speech Module** | STT (Google / Sphinx / Whisper) + TTS (pyttsx3 / gTTS / ElevenLabs) |
| 🔌 **Plugin System** | Register, load, and dispatch custom skills from files or directories |
| 🔁 **Auto-retry** | Exponential back-off on transient API errors, built-in rate limiting |
| 🖥️ **Interactive REPL** | Built-in terminal chat loop with optional voice mode |

---

## Installation

### Minimal (core only)

```bash
pip install wizardai
# or
pip install -r requirements.txt
```

### With specific features

```bash
# OpenAI backend
pip install "wizardai[openai]"

# Anthropic backend
pip install "wizardai[anthropic]"

# Computer vision
pip install "wizardai[vision]"

# Speech (STT + TTS)
pip install "wizardai[speech]"

# High-quality offline STT (Whisper)
pip install "wizardai[whisper]"

# Everything
pip install "wizardai[all]"
```

### From source

```bash
git clone https://github.com/yourusername/wizardai-sdk.git
cd wizardai-sdk
pip install -e ".[all]"
```

---

## Quick Start

```python
import wizardai

# One-line setup with OpenAI
with wizardai.WizardAI(openai_api_key="sk-...") as wiz:

    # Pattern-based chat (no API call)
    wiz.agent.add_pattern("hello *", "Hello there, {wildcard}!")
    print(wiz.chat("hello world"))           # → "Hello there, world!"

    # LLM call
    print(wiz.ask("What is the speed of light?"))

    # Long-term memory
    wiz.remember("user_name", "Alice")
    print(wiz.recall("user_name"))           # → "Alice"
```

---

## Modules

### WizardAI Core

`WizardAI` is the top-level orchestrator. It wires all sub-modules together and exposes convenience shortcuts for the most common tasks.

```python
from wizardai import WizardAI

wiz = WizardAI(
    ai_backend="openai",          # "openai" | "anthropic" | "huggingface" | "custom"
    openai_api_key="sk-...",
    default_model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.7,
    enable_vision=True,           # open webcam on start()
    enable_speech=True,           # init STT/TTS on start()
    stt_backend="google",
    tts_backend="pyttsx3",
    agent_name="WizardBot",
    fallback_response="I'm not sure about that.",
    max_history=50,
    memory_path="session.json",   # persist memory to disk
    system_prompt="You are a helpful assistant.",
    log_level="INFO",
    data_dir="./wizardai_data",
)

wiz.start()   # opens camera, initialises speech engine, notifies plugins

# --- Chat shortcuts ---
reply = wiz.chat("hello")          # tries patterns first, then LLM
reply = wiz.ask("Tell me a joke")  # always calls the LLM

# --- Voice shortcuts ---
wiz.say("Hello from WizardAI!")    # TTS
text = wiz.listen(timeout=5)       # STT
reply = wiz.voice_chat()           # listen → chat → say

# --- Vision shortcuts ---
frame = wiz.capture()              # single frame (numpy ndarray)
path  = wiz.snapshot("photo.jpg") # capture + save

# --- Memory shortcuts ---
wiz.remember("city", "Paris")
city = wiz.recall("city")         # → "Paris"
history = wiz.get_history(n=5)    # last 5 turns as list of dicts

wiz.stop()

# Use as a context manager
with WizardAI(openai_api_key="sk-...") as wiz:
    print(wiz.ask("Hi!"))

# Launch interactive terminal REPL
wiz.run_repl()
wiz.run_repl(voice_mode=True)      # voice I/O
```

---

### AIClient

Unified interface for all supported LLM backends. Handles retries, rate limiting, and key resolution from environment variables automatically.

**Supported backends**

| Backend | Default model | Env var |
|---|---|---|
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` |
| `huggingface` | `mistralai/Mistral-7B-Instruct-v0.2` | `HUGGINGFACE_API_KEY` |
| `custom` | `default` | `WIZARDAI_CUSTOM_API_KEY` |

```python
from wizardai import AIClient, AIBackend

# --- OpenAI ---
client = AIClient(backend="openai", api_key="sk-...")

# Single-turn completion
resp = client.complete("Write a haiku about Python.")
print(resp.text)
print(resp.usage)       # {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
print(resp.latency_ms)  # round-trip time in ms

# Multi-turn chat
messages = [
    {"role": "user",      "content": "My name is Bob."},
    {"role": "assistant", "content": "Nice to meet you, Bob!"},
    {"role": "user",      "content": "What is my name?"},
]
resp = client.chat(messages, system_prompt="You are a helpful assistant.")
print(resp.text)  # → "Your name is Bob."

# --- Anthropic ---
client = AIClient(backend=AIBackend.ANTHROPIC, api_key="sk-ant-...")
resp = client.chat(messages, model="claude-opus-4-5")

# --- HuggingFace ---
client = AIClient(backend="huggingface", api_key="hf-...")
resp = client.complete("Explain recursion.", model="mistralai/Mistral-7B-Instruct-v0.2")

# --- Custom endpoint (OpenAI-compatible) ---
client = AIClient(
    backend="custom",
    endpoint="https://my-llm.example.com/v1/chat/completions",
    api_key="secret",
)
resp = client.complete("Hello!")

# Runtime config changes
client.set_model("gpt-4o")
client.set_api_key("sk-new-key...")
```

---

### ConversationAgent

AIML-style rule-based chat engine with wildcard patterns, priorities, context rules, and callable templates — with memory integration.

```python
from wizardai import ConversationAgent, Pattern, MemoryManager

mem   = MemoryManager()
agent = ConversationAgent(name="Aria", fallback="I don't know!", memory=mem)

# --- Simple patterns ---
agent.add_pattern("hello", "Hi there!")
agent.add_pattern("what is your name", "I'm Aria, your AI assistant.")

# --- Wildcards ---
#   *   matches any sequence of words → {wildcard}
#   ?   matches exactly one word
agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")
agent.add_pattern("what is ? plus ?", "Let me calculate that for you.")

# --- Callable template (dynamic response) ---
import random
agent.add_pattern(
    "tell me a joke",
    lambda text, ctx: random.choice([
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "I told a joke about UDP once. I don't know if you got it.",
    ]),
)

# --- Priority (higher wins when patterns overlap) ---
agent.add_pattern("hello world", "Special hello-world greeting!", priority=10)

# --- Context-aware patterns ---
agent.add_pattern("yes", "Great, let's proceed!",  context="confirm_action")
agent.add_pattern("no",  "OK, I'll cancel that.",  context="confirm_action")
agent.set_context("confirm_action")

# --- Pattern object (full control) ---
agent.add_pattern_obj(Pattern(
    pattern="weather in *",
    template="Checking weather for {wildcard}…",
    priority=5,
    tags=["weather", "location"],
))

# Chat
print(agent.respond("hello"))               # → "Hi there!"
print(agent.respond("my name is Charlie"))  # → "Nice to meet you, Charlie!"
print(agent.respond("tell me a joke"))      # → random joke

# Introspection
agent.list_patterns()           # list all Pattern objects
agent.remove_pattern("hello")   # remove by pattern string
agent.clear_patterns()          # wipe all rules
```

---

### MemoryManager

Provides a sliding-window conversation history (short-term) plus a persistent key-value store (long-term), with optional JSON disk persistence.

```python
from wizardai import MemoryManager

mem = MemoryManager(
    max_history=20,           # keep last 20 messages
    persist_path="mem.json",  # auto-save on every write
)

# --- Short-term (conversation history) ---
mem.add_message("user",      "What's the capital of France?")
mem.add_message("assistant", "Paris!")

history = mem.get_history()                # list of Message objects
history = mem.get_history(n=5)            # last 5 messages
history = mem.get_history(role_filter="user")  # only user turns

# Ready-to-send format for APIs
api_msgs = mem.get_messages_for_api()     # [{"role": …, "content": …}, …]

last = mem.last_message()                 # most recent Message
last = mem.last_message(role="user")

results = mem.search_history("France", top_k=3)  # (Message, score) tuples
mem.clear_history()

# --- Long-term memory ---
mem.remember("user_name", "Alice")
mem.remember("preferences", {"theme": "dark", "lang": "en"})

name  = mem.recall("user_name")           # → "Alice"
prefs = mem.recall("preferences")         # → dict
mem.forget("user_name")
keys  = mem.list_memories()               # all long-term keys

# --- Ephemeral session context (not persisted) ---
mem.set_context("current_topic", "weather")
topic = mem.get_context("current_topic")
mem.clear_context()

# --- Persistence ---
mem.save("backup.json")
mem.load("backup.json")
```

---

### VisionModule

Real-time webcam capture and image processing powered by OpenCV. All OpenCV imports are deferred — the rest of WizardAI works even if `opencv-python` is not installed.

```python
from wizardai import VisionModule

cam = VisionModule(
    device_id=0,    # 0 = default webcam
    width=1280,
    height=720,
    fps=30,
)
cam.open()

# --- Capture ---
frame = cam.capture_frame()              # numpy ndarray (BGR)
cam.save_frame(frame, "snapshot.jpg")   # save to disk

# --- Encode for vision-capable LLMs ---
b64 = cam.encode_to_base64(frame)       # base64 JPEG string
# → pass as image_b64= to wiz.ask(...)

# --- Display ---
cam.show_frame(frame, window="Preview") # OpenCV window

# --- Detection ---
faces = cam.detect_faces(frame)
# → [{"label": "face", "confidence": 1.0, "bbox": (x, y, w, h)}, …]

# --- Streaming (background thread) ---
def on_frame(frame):
    faces = cam.detect_faces(frame)
    if faces:
        print(f"Detected {len(faces)} face(s)")

cam.start_stream(callback=on_frame, show_preview=True)
import time; time.sleep(10)
cam.stop_stream()

# --- Filters ---
gray  = cam.to_grayscale(frame)
small = cam.resize(frame, width=320)

cam.close()

# Context manager
with VisionModule() as cam:
    frame = cam.capture_frame()
    cam.save_frame(frame, "out.jpg")
```

---

### SpeechModule

Speech recognition (STT) and text-to-speech (TTS) with multiple backend options.

**STT backends**

| Backend | Connectivity | Package needed |
|---|---|---|
| `google` | Online | `SpeechRecognition` |
| `sphinx` | Offline | `SpeechRecognition`, `pocketsphinx` |
| `whisper` | Offline | `openai-whisper`, `numpy` |

**TTS backends**

| Backend | Connectivity | Package needed |
|---|---|---|
| `pyttsx3` | Offline | `pyttsx3` |
| `gtts` | Online | `gtts`, `pygame` |
| `elevenlabs` | Online | `requests` + API key |

```python
from wizardai import SpeechModule

speech = SpeechModule(
    stt_backend="google",      # or "sphinx", "whisper"
    tts_backend="pyttsx3",     # or "gtts", "elevenlabs"
    language="en-US",
    tts_rate=150,              # words per minute (pyttsx3)
    tts_volume=1.0,
    elevenlabs_api_key="...",  # only for elevenlabs TTS
)
speech.init_tts()

# --- TTS ---
speech.say("Hello, I am WizardAI!")
speech.say("This plays in background.", blocking=False)
speech.save_audio("Hi there!", "greeting.mp3")  # save to file

# --- STT ---
text = speech.listen(timeout=5.0)  # returns None on silence / error
print("You said:", text)

# transcribe from a file
text = speech.transcribe_file("audio.wav")

# --- Streaming TTS (word-by-word) ---
for word in speech.stream_say("Generating token by token…"):
    print(word, end=" ", flush=True)

# --- Microphone listing ---
mics = speech.list_microphones()
for i, name in enumerate(mics):
    print(i, name)
```

---

### Plugin System

Extend WizardAI with custom skills by subclassing `PluginBase` and registering with `PluginManager`.

#### Creating a plugin

```python
from wizardai import PluginBase
from typing import Optional

class WeatherPlugin(PluginBase):
    name        = "weather"
    description = "Returns mock weather data for any city."
    version     = "1.0.0"
    triggers    = ["weather in *", "what's the weather in *"]

    def setup(self):
        # called once after __init__ — initialise resources here
        self.api_key = self.config.get("api_key", "")

    def on_message(self, text: str, context: dict) -> Optional[str]:
        city = text.split("in", 1)[-1].strip()
        return f"The weather in {city} is sunny, 25 °C."

    def on_start(self):
        self.logger.info("WeatherPlugin session started.")

    def on_stop(self):
        self.logger.info("WeatherPlugin session ended.")

    def teardown(self):
        # called when unregistered
        pass
```

#### Using the PluginManager

```python
from wizardai import PluginManager

manager = PluginManager()

# Register a class
manager.register(WeatherPlugin, config={"api_key": "abc123"})

# Dispatch to first matching plugin
response = manager.dispatch("weather in Paris", context={})
print(response)   # → "The weather in Paris is sunny, 25 °C."

# Dispatch to ALL matching plugins
results = manager.dispatch_all("weather in London")
# → [("weather", "The weather in London is sunny, 25 °C.")]

# List, enable/disable
for plugin in manager.list_plugins():
    print(plugin)

manager.get("weather").disable()
manager.get("weather").enable()

# Load from files
manager.load_from_file("my_plugin.py", config={})
manager.load_from_directory("./plugins/", config={})

# Lifecycle hooks
manager.start_all()   # calls on_start() on all enabled plugins
manager.stop_all()    # calls on_stop()

manager.unregister("weather")

# Register via WizardAI core
wiz.add_plugin(WeatherPlugin, config={"api_key": "abc123"})
wiz.load_plugins_from_dir("./plugins/")
```

---

### Exceptions

All exceptions inherit from `WizardAIError`.

```python
from wizardai import (
    WizardAIError,
    APIError,
    VisionError,
    SpeechError,
    ConversationError,
)
from wizardai.exceptions import (
    RateLimitError,
    AuthenticationError,
    CameraNotFoundError,
    MicrophoneNotFoundError,
    PluginError,
    ConfigurationError,
)

try:
    reply = wiz.ask("Hello!")
except AuthenticationError as e:
    print("Bad API key:", e.backend)
except RateLimitError as e:
    print("Slow down! Retry after:", e.retry_after)
except APIError as e:
    print(f"API error {e.code}:", e.message)
except VisionError:
    print("Camera problem")
except SpeechError:
    print("Microphone or TTS problem")
except WizardAIError as e:
    print("General WizardAI error:", e)
```

---

## Folder Structure

```
wizardai-sdk/
│
├── wizardai/                   # Main package
│   ├── __init__.py             # Public API surface
│   ├── core.py                 # WizardAI orchestrator
│   ├── ai_client.py            # Multi-backend LLM client
│   ├── conversation.py         # ConversationAgent + Pattern
│   ├── memory.py               # MemoryManager + Message
│   ├── vision.py               # VisionModule (OpenCV)
│   ├── speech.py               # SpeechModule (STT/TTS)
│   ├── plugins.py              # PluginBase + PluginManager
│   ├── utils.py                # Logger, FileHelper, DataSerializer, RateLimiter
│   └── exceptions.py           # Custom exception hierarchy
│
├── examples/
│   └── full_demo.py            # Complete walkthrough of every module
│
├── plugins/                    # Drop-in custom plugin files
│   └── sample_plugin.py
│
├── tests/
│   ├── __init__.py
│   ├── test_ai_client.py
│   ├── test_conversation.py
│   ├── test_memory.py
│   ├── test_plugins.py
│   └── test_core.py
│
├── docs/                       # Additional documentation / assets
│   └── architecture.png
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
├── requirements.txt            # Minimal: requests only
└── requirements-full.txt       # All optional dependencies
```

---

## Configuration Reference

### WizardAI constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ai_backend` | `str` | `"openai"` | LLM backend to use |
| `openai_api_key` | `str` | `None` | OpenAI API key (env: `OPENAI_API_KEY`) |
| `anthropic_api_key` | `str` | `None` | Anthropic key (env: `ANTHROPIC_API_KEY`) |
| `huggingface_api_key` | `str` | `None` | HuggingFace key (env: `HUGGINGFACE_API_KEY`) |
| `custom_endpoint` | `str` | `None` | URL for a custom REST endpoint |
| `default_model` | `str` | `None` | Override default model |
| `max_tokens` | `int` | `1024` | Max tokens per LLM response |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `enable_vision` | `bool` | `False` | Open webcam on `start()` |
| `camera_device` | `int` | `0` | OpenCV camera index |
| `camera_width` | `int` | `640` | Capture width (px) |
| `camera_height` | `int` | `480` | Capture height (px) |
| `enable_speech` | `bool` | `False` | Init STT/TTS on `start()` |
| `stt_backend` | `str` | `"google"` | STT backend |
| `tts_backend` | `str` | `"pyttsx3"` | TTS engine |
| `language` | `str` | `"en-US"` | BCP-47 language code |
| `agent_name` | `str` | `"WizardBot"` | Agent display name |
| `fallback_response` | `str` | `"I'm not sure…"` | Default when no pattern matches |
| `max_history` | `int` | `50` | Short-term memory window |
| `memory_path` | `str` | `None` | Path for memory persistence |
| `system_prompt` | `str` | `None` | Default LLM system prompt |
| `log_level` | `str` | `"INFO"` | `DEBUG \| INFO \| WARNING \| ERROR` |
| `log_file` | `str` | `None` | Optional log file path |
| `data_dir` | `str` | `"./wizardai_data"` | Working data directory |

### Environment variables

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export HUGGINGFACE_API_KEY="hf-..."
export WIZARDAI_CUSTOM_API_KEY="..."
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Format code: `black . && isort .`
6. Open a pull request.

---

## License

MIT © WizardAI Contributors — see [LICENSE](LICENSE) for full text.
