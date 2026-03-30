"""
WizardAI Speech Module
----------------------
Speech recognition (STT) and text-to-speech (TTS) capabilities.
Supports multiple recogniser backends (Google, Sphinx, Whisper) and
TTS engines (pyttsx3, gTTS, ElevenLabs stub).
"""

from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .exceptions import MicrophoneNotFoundError, SpeechError
from .utils import Logger


# ---------------------------------------------------------------------------
# SpeechModule
# ---------------------------------------------------------------------------

class SpeechModule:
    """Handles speech recognition (STT) and text-to-speech (TTS).

    Supports multiple STT backends:
    - ``google``  – Google Web Speech API (requires internet, free).
    - ``sphinx``  – CMU PocketSphinx (offline, requires ``pocketsphinx``).
    - ``whisper`` – OpenAI Whisper (offline, high accuracy, requires GPU
                    recommended but CPU works).

    Supports multiple TTS backends:
    - ``pyttsx3`` – offline, built-in OS voices.
    - ``gtts``    – Google TTS (requires internet, produces mp3).
    - ``elevenlabs`` – ElevenLabs API (high-quality, requires API key).

    Example::

        speech = SpeechModule(stt_backend="google", tts_backend="pyttsx3")

        # Listen and transcribe
        text = speech.listen(timeout=5)
        print("You said:", text)

        # Speak a response
        speech.say("Hello! I heard: " + text)
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
        """
        Args:
            stt_backend:          STT engine: 'google' | 'sphinx' | 'whisper'.
            tts_backend:          TTS engine: 'pyttsx3' | 'gtts' | 'elevenlabs'.
            language:             BCP-47 language code (e.g. 'en-US', 'fr-FR').
            tts_rate:             pyttsx3 words-per-minute rate.
            tts_volume:           pyttsx3 volume (0.0–1.0).
            elevenlabs_api_key:   ElevenLabs API key (env: ELEVENLABS_API_KEY).
            elevenlabs_voice_id:  ElevenLabs voice ID.
            logger:               Optional Logger instance.
        """
        self.stt_backend = stt_backend.lower()
        self.tts_backend = tts_backend.lower()
        self.language = language
        self.tts_rate = tts_rate
        self.tts_volume = tts_volume
        self.elevenlabs_api_key = elevenlabs_api_key or os.environ.get(
            "ELEVENLABS_API_KEY", ""
        )
        self.elevenlabs_voice_id = elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"
        self.logger = logger or Logger("SpeechModule")

        self._recogniser = None
        self._tts_engine = None
        self._whisper_model = None
        self._listening = threading.Event()
        self._continuous_callbacks: List[Callable[[str], None]] = []

        self._init_tts()
        self.logger.info(
            f"SpeechModule: STT={self.stt_backend}, TTS={self.tts_backend}, "
            f"lang={self.language}"
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _get_recogniser(self):
        """Lazily initialise the speech_recognition Recognizer."""
        if self._recogniser is None:
            try:
                import speech_recognition as sr
                self._recogniser = sr.Recognizer()
            except ImportError:
                raise SpeechError(
                    "The 'SpeechRecognition' package is required for STT. "
                    "Install it with: pip install SpeechRecognition"
                )
        return self._recogniser

    def _init_tts(self):
        """Lazily initialise the TTS engine for pyttsx3."""
        if self.tts_backend == "pyttsx3":
            try:
                import pyttsx3
                self._tts_engine = pyttsx3.init()
                self._tts_engine.setProperty("rate", self.tts_rate)
                self._tts_engine.setProperty("volume", self.tts_volume)
            except ImportError:
                self.logger.warning(
                    "pyttsx3 not installed. TTS will be unavailable. "
                    "Install: pip install pyttsx3"
                )
            except Exception as exc:
                self.logger.warning(f"pyttsx3 init error: {exc}")

    # ------------------------------------------------------------------
    # Speech-to-Text (STT)
    # ------------------------------------------------------------------

    def listen(
        self,
        timeout: Optional[float] = 5.0,
        phrase_time_limit: Optional[float] = 15.0,
        adjust_noise: bool = True,
        device_index: Optional[int] = None,
    ) -> str:
        """Record audio from the microphone and transcribe it.

        Args:
            timeout:          Seconds to wait for speech to start; None = wait
                              indefinitely.
            phrase_time_limit: Maximum seconds to record per phrase.
            adjust_noise:     If True, calibrate the recogniser for ambient
                              noise before listening.
            device_index:     Microphone device index (None = default).

        Returns:
            Transcribed text string.

        Raises:
            SpeechError: If recognition fails or no speech is detected.
            MicrophoneNotFoundError: If no microphone is available.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            raise SpeechError(
                "The 'SpeechRecognition' package is required. "
                "pip install SpeechRecognition"
            )

        recogniser = self._get_recogniser()

        try:
            mic = sr.Microphone(device_index=device_index)
        except AttributeError:
            raise MicrophoneNotFoundError()
        except OSError:
            raise MicrophoneNotFoundError()

        with mic as source:
            if adjust_noise:
                self.logger.debug("Adjusting for ambient noise…")
                recogniser.adjust_for_ambient_noise(source, duration=0.5)
            self.logger.debug("Listening…")
            try:
                audio = recogniser.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
            except sr.WaitTimeoutError:
                raise SpeechError("No speech detected within timeout.")

        return self._transcribe(audio)

    def transcribe_file(self, path: Union[str, Path]) -> str:
        """Transcribe an audio file to text.

        Args:
            path: Path to an audio file (WAV, AIFF, FLAC, …).

        Returns:
            Transcribed text.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            raise SpeechError("SpeechRecognition required. pip install SpeechRecognition")

        recogniser = self._get_recogniser()
        with sr.AudioFile(str(path)) as source:
            audio = recogniser.record(source)
        return self._transcribe(audio)

    def _transcribe(self, audio) -> str:
        """Route the audio to the configured STT backend."""
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
        """Transcribe using OpenAI Whisper (local model)."""
        try:
            import whisper
            import numpy as np
        except ImportError:
            raise SpeechError(
                "Whisper STT requires 'openai-whisper' and 'numpy'. "
                "pip install openai-whisper numpy"
            )

        if self._whisper_model is None:
            self.logger.info("Loading Whisper model (this may take a moment)…")
            self._whisper_model = whisper.load_model("base")

        # Convert AudioData to float32 numpy array
        raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._whisper_model.transcribe(audio_np, language=self.language[:2])
        return result.get("text", "").strip()

    # ------------------------------------------------------------------
    # Text-to-Speech (TTS)
    # ------------------------------------------------------------------

    def say(self, text: str, blocking: bool = True) -> Optional[str]:
        """Convert *text* to speech and play or save it.

        Args:
            text:     Text to speak.
            blocking: If True, wait for playback to finish (pyttsx3 only).

        Returns:
            For 'gtts' backend, returns the path to the generated mp3.
            Otherwise None.

        Raises:
            SpeechError: If TTS fails.
        """
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
        """Convert *text* to speech and save to an audio file.

        Args:
            text: Text to synthesise.
            path: Destination file path (.mp3 or .wav depending on backend).

        Returns:
            The path that was written.
        """
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
            tts = gTTS(text=text, lang=self.language[:2])
            tts.save(str(p))
        elif self.tts_backend == "elevenlabs":
            audio_bytes = self._elevenlabs_synthesise(text)
            p.write_bytes(audio_bytes)
        else:
            raise SpeechError(f"Unknown TTS backend: {self.tts_backend!r}")

        self.logger.debug(f"[TTS] Saved to {p}")
        return p

    # ------------------------------------------------------------------
    # TTS backend implementations
    # ------------------------------------------------------------------

    def _say_pyttsx3(self, text: str, blocking: bool):
        """Speak using pyttsx3 (offline)."""
        if self._tts_engine is None:
            raise SpeechError(
                "pyttsx3 engine not available. "
                "Install: pip install pyttsx3"
            )
        self._tts_engine.say(text)
        if blocking:
            self._tts_engine.runAndWait()

    def _say_gtts(self, text: str) -> str:
        """Speak using gTTS (online, requires internet)."""
        try:
            from gtts import gTTS
        except ImportError:
            raise SpeechError("gTTS required. pip install gtts")

        try:
            import pygame
        except ImportError:
            pygame = None  # will fall back to saving only

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
                    import time; time.sleep(0.1)
                pygame.mixer.music.unload()
            except Exception as exc:
                self.logger.warning(f"pygame playback failed: {exc}")

        return tmp_path

    def _say_elevenlabs(self, text: str) -> str:
        """Speak using ElevenLabs API."""
        audio_bytes = self._elevenlabs_synthesise(text)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            import time
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except ImportError:
            self.logger.warning(
                "pygame not installed. Audio saved to temp file but not played. "
                "pip install pygame"
            )
        return tmp_path

    def _elevenlabs_synthesise(self, text: str) -> bytes:
        """Call ElevenLabs TTS API and return raw audio bytes."""
        if not self.elevenlabs_api_key:
            raise SpeechError(
                "ElevenLabs API key is required. "
                "Set env var ELEVENLABS_API_KEY or pass elevenlabs_api_key=."
            )
        try:
            import requests
        except ImportError:
            raise SpeechError("requests required. pip install requests")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}"
        headers = {
            "xi-api-key": self.elevenlabs_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if not r.ok:
            raise SpeechError(f"ElevenLabs error {r.status_code}: {r.text}")
        return r.content

    # ------------------------------------------------------------------
    # Continuous listening
    # ------------------------------------------------------------------

    def add_listener(self, callback: Callable[[str], None]):
        """Register a callback for continuous listening mode.

        The callback receives each transcribed utterance as a string.

        Args:
            callback: ``fn(text: str)`` called for every utterance.
        """
        self._continuous_callbacks.append(callback)

    def start_continuous_listening(
        self,
        callback: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
        phrase_time_limit: float = 10.0,
    ):
        """Start listening for speech in a background thread.

        All registered callbacks (plus *callback* if provided) are invoked
        with each transcribed utterance.

        Args:
            callback:         Optional additional per-utterance callback.
            timeout:          Seconds to wait for each utterance (None = block).
            phrase_time_limit: Max seconds per utterance.
        """
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
        """Stop the background listening thread."""
        self._listening.clear()
        self.logger.info("Continuous listening stopped.")

    def _continuous_loop(self, timeout, phrase_time_limit):
        """Internal loop for continuous listening (runs in a daemon thread)."""
        while self._listening.is_set():
            try:
                text = self.listen(
                    timeout=timeout, phrase_time_limit=phrase_time_limit
                )
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
    # Utility
    # ------------------------------------------------------------------

    def list_microphones(self) -> List[Dict[str, Union[int, str]]]:
        """Return a list of available microphone devices.

        Returns:
            List of dicts with 'index' and 'name' keys.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            raise SpeechError("SpeechRecognition required. pip install SpeechRecognition")

        mics = []
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            mics.append({"index": i, "name": name})
        return mics

    def set_tts_rate(self, rate: int):
        """Set the pyttsx3 speech rate (words per minute)."""
        self.tts_rate = rate
        if self._tts_engine:
            self._tts_engine.setProperty("rate", rate)

    def set_tts_volume(self, volume: float):
        """Set the pyttsx3 TTS volume (0.0–1.0)."""
        self.tts_volume = max(0.0, min(1.0, volume))
        if self._tts_engine:
            self._tts_engine.setProperty("volume", self.tts_volume)

    def set_tts_voice(self, voice_id: str):
        """Set the pyttsx3 voice by its ID."""
        if self._tts_engine:
            self._tts_engine.setProperty("voice", voice_id)

    def list_voices(self) -> List[Dict]:
        """Return available pyttsx3 voices."""
        if not self._tts_engine:
            return []
        return [
            {"id": v.id, "name": v.name, "languages": v.languages}
            for v in self._tts_engine.getProperty("voices")
        ]

    def __repr__(self):
        return (
            f"SpeechModule(stt={self.stt_backend!r}, tts={self.tts_backend!r}, "
            f"lang={self.language!r})"
        )
