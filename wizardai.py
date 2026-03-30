"""
WizardAI - A powerful, all-in-one Python SDK for AI integration.

Combines conversational AI, computer vision, speech I/O, and more
in a single easy-to-use module.

Author: WizardAI Contributors
Version: 1.0.0
License: MIT
"""

from .core import WizardAI
from .ai_client import AIClient, AIBackend
from .vision import VisionModule
from .speech import SpeechModule
from .conversation import ConversationAgent, Pattern
from .memory import MemoryManager
from .plugins import PluginBase, PluginManager
from .utils import Logger, FileHelper, DataSerializer
from .exceptions import (
    WizardAIError,
    APIError,
    VisionError,
    SpeechError,
    ConversationError,
)

__version__ = "1.0.0"
__author__ = "WizardAI Contributors"
__license__ = "MIT"

__all__ = [
    "WizardAI",
    "AIClient",
    "AIBackend",
    "VisionModule",
    "SpeechModule",
    "ConversationAgent",
    "Pattern",
    "MemoryManager",
    "PluginBase",
    "PluginManager",
    "Logger",
    "FileHelper",
    "DataSerializer",
    "WizardAIError",
    "APIError",
    "VisionError",
    "SpeechError",
    "ConversationError",
]
