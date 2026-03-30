"""
WizardAI Custom Exceptions
--------------------------
Defines all exception types used throughout the WizardAI SDK.
"""


class WizardAIError(Exception):
    """Base exception class for all WizardAI errors."""

    def __init__(self, message: str, code: int = None):
        super().__init__(message)
        self.message = message
        self.code = code

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, code={self.code})"


class APIError(WizardAIError):
    """Raised when an AI API call fails.

    Attributes:
        message: Human-readable error description.
        code: HTTP status code or API error code.
        backend: The AI backend that raised the error.
    """

    def __init__(self, message: str, code: int = None, backend: str = None):
        super().__init__(message, code)
        self.backend = backend

    def __repr__(self):
        return (
            f"APIError(message={self.message!r}, "
            f"code={self.code}, backend={self.backend!r})"
        )


class RateLimitError(APIError):
    """Raised when an API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float = None):
        super().__init__(message, code=429)
        self.retry_after = retry_after


class VisionError(WizardAIError):
    """Raised when a camera or image processing operation fails."""
    pass


class CameraNotFoundError(VisionError):
    """Raised when the requested camera device is not found."""

    def __init__(self, device_id: int = 0):
        super().__init__(f"Camera device {device_id} not found or unavailable.")
        self.device_id = device_id


class SpeechError(WizardAIError):
    """Raised when a speech recognition or TTS operation fails."""
    pass


class MicrophoneNotFoundError(SpeechError):
    """Raised when no microphone is detected."""

    def __init__(self):
        super().__init__("No microphone device found. Please check your audio input.")


class ConversationError(WizardAIError):
    """Raised when the conversation engine encounters an error."""
    pass


class PluginError(WizardAIError):
    """Raised when a plugin fails to load or execute."""

    def __init__(self, message: str, plugin_name: str = None):
        super().__init__(message)
        self.plugin_name = plugin_name


class ConfigurationError(WizardAIError):
    """Raised when the SDK is misconfigured."""
    pass


class AuthenticationError(APIError):
    """Raised when API key validation fails."""

    def __init__(self, backend: str = None):
        super().__init__(
            f"Authentication failed for backend: {backend or 'unknown'}. "
            "Please check your API key.",
            code=401,
            backend=backend,
        )
