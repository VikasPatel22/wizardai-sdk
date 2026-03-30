"""
WizardAI Utilities
------------------
Common utilities: logging, file I/O helpers, and data serialization.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    """Configurable logger for WizardAI components.

    Wraps Python's standard ``logging`` module with sensible defaults and
    coloured console output (when a terminal supports it).

    Example::

        log = Logger("my_app", level="DEBUG")
        log.info("WizardAI started")
        log.warning("Low memory")
        log.error("Something went wrong")
    """

    _COLOURS = {
        "DEBUG": "\033[94m",     # blue
        "INFO": "\033[92m",      # green
        "WARNING": "\033[93m",   # yellow
        "ERROR": "\033[91m",     # red
        "CRITICAL": "\033[95m",  # magenta
        "RESET": "\033[0m",
    }

    def __init__(
        self,
        name: str = "wizardai",
        level: str = "INFO",
        log_file: Optional[str] = None,
        coloured: bool = True,
    ):
        """Initialise the logger.

        Args:
            name:      Logger name (shown in output).
            level:     Minimum log level: DEBUG | INFO | WARNING | ERROR | CRITICAL.
            log_file:  Optional path to write logs to disk.
            coloured:  Enable ANSI colour codes in console output.
        """
        self.name = name
        self.coloured = coloured
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self._build_formatter())
            self._logger.addHandler(handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self._logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_formatter(self) -> logging.Formatter:
        if self.coloured:

            class ColouredFormatter(logging.Formatter):
                _C = Logger._COLOURS

                def format(self, record):  # noqa: A003
                    colour = self._C.get(record.levelname, "")
                    reset = self._C["RESET"]
                    record.levelname = f"{colour}{record.levelname}{reset}"
                    return super().format(record)

            return ColouredFormatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )

        return logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def debug(self, msg: str, *args, **kwargs):
        """Log a DEBUG message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log an INFO message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log a WARNING message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log an ERROR message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log a CRITICAL message."""
        self._logger.critical(msg, *args, **kwargs)

    def set_level(self, level: str):
        """Dynamically change the log level."""
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# FileHelper
# ---------------------------------------------------------------------------

class FileHelper:
    """High-level file I/O helpers used across WizardAI.

    Example::

        fh = FileHelper(base_dir="./data")
        fh.write_text("hello.txt", "Hello, world!")
        content = fh.read_text("hello.txt")
        fh.ensure_dir("models/cache")
    """

    def __init__(self, base_dir: Union[str, Path] = "."):
        """
        Args:
            base_dir: Base directory all relative paths are resolved against.
        """
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def resolve(self, path: Union[str, Path]) -> Path:
        """Resolve *path* relative to *base_dir* if it is not absolute."""
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """Create *path* (and parents) if it does not exist; return the Path."""
        full = self.resolve(path)
        full.mkdir(parents=True, exist_ok=True)
        return full

    # ------------------------------------------------------------------
    # Text I/O
    # ------------------------------------------------------------------

    def write_text(
        self,
        path: Union[str, Path],
        content: str,
        encoding: str = "utf-8",
        append: bool = False,
    ) -> Path:
        """Write *content* to a text file.

        Args:
            path:     File path (relative to base_dir or absolute).
            content:  String to write.
            encoding: File encoding (default utf-8).
            append:   If True, append instead of overwrite.

        Returns:
            The resolved Path that was written.
        """
        full = self.resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(full, mode, encoding=encoding) as fh:
            fh.write(content)
        return full

    def read_text(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> str:
        """Read and return the contents of a text file."""
        with open(self.resolve(path), "r", encoding=encoding) as fh:
            return fh.read()

    def read_lines(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        strip: bool = True,
    ) -> List[str]:
        """Return file contents as a list of lines."""
        lines = self.read_text(path, encoding=encoding).splitlines()
        return [ln.strip() for ln in lines] if strip else lines

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def write_json(
        self,
        path: Union[str, Path],
        data: Any,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> Path:
        """Serialise *data* to a JSON file."""
        full = self.resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=ensure_ascii)
        return full

    def read_json(self, path: Union[str, Path]) -> Any:
        """Deserialise a JSON file and return the Python object."""
        with open(self.resolve(path), "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def write_csv(
        self,
        path: Union[str, Path],
        rows: List[Dict],
        fieldnames: Optional[List[str]] = None,
    ) -> Path:
        """Write a list of dicts to a CSV file."""
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
        """Read a CSV file and return a list of dicts."""
        with open(self.resolve(path), "r", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def copy(self, src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """Copy a file from *src* to *dst*."""
        dst_path = self.resolve(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(self.resolve(src)), str(dst_path))
        return dst_path

    def delete(self, path: Union[str, Path]) -> bool:
        """Delete a file if it exists; return True if deleted."""
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
        """Return a list of files matching *pattern* in *directory*."""
        full = self.resolve(directory)
        if recursive:
            return list(full.rglob(pattern))
        return list(full.glob(pattern))

    def timestamp_filename(self, name: str, ext: str = "") -> str:
        """Generate a filename with the current timestamp embedded.

        Example::

            fh.timestamp_filename("log", ".txt")
            # => "log_20240315_142305.txt"
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ext if ext.startswith(".") or not ext else f".{ext}"
        return f"{name}_{ts}{suffix}"


# ---------------------------------------------------------------------------
# DataSerializer
# ---------------------------------------------------------------------------

class DataSerializer:
    """Unified data serialization supporting JSON, Pickle, and gzip variants.

    Example::

        ds = DataSerializer()
        ds.save({"model": "gpt-4", "tokens": 512}, "config.json")
        ds.save(large_object, "model.pkl.gz", compress=True)
        cfg = ds.load("config.json")
    """

    # ------------------------------------------------------------------
    # Detect format
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(path: Union[str, Path]) -> str:
        """Infer format from file extension: json | pickle | compressed."""
        name = str(path).lower()
        if name.endswith(".json"):
            return "json"
        if name.endswith(".json.gz"):
            return "json.gz"
        if name.endswith((".pkl", ".pickle")):
            return "pickle"
        if name.endswith((".pkl.gz", ".pickle.gz")):
            return "pickle.gz"
        return "json"  # default

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        data: Any,
        path: Union[str, Path],
        compress: bool = False,
        indent: int = 2,
    ) -> Path:
        """Serialise *data* to *path*.

        The format is chosen from the file extension:
        ``*.json`` → JSON text, ``*.pkl`` / ``*.pickle`` → Pickle binary.
        Append ``.gz`` to either for gzip compression.

        Args:
            data:     Python object to serialise.
            path:     Destination file path.
            compress: Force gzip compression regardless of extension.
            indent:   JSON indentation level.

        Returns:
            The resolved Path that was written.
        """
        p = Path(path)
        fmt = self._detect_format(p)
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
            raise ValueError(f"Unsupported serialization format for path: {path}")

        return p

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, path: Union[str, Path]) -> Any:
        """Deserialise and return data from *path*.

        Format is inferred from the file extension (see :meth:`save`).

        Args:
            path: Source file path.

        Returns:
            The deserialised Python object.
        """
        p = Path(path)
        fmt = self._detect_format(p)

        if fmt == "json":
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        elif fmt == "json.gz":
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        elif fmt == "pickle":
            with open(p, "rb") as fh:
                return pickle.load(fh)
        elif fmt == "pickle.gz":
            with gzip.open(p, "rb") as fh:
                return pickle.load(fh)
        else:
            raise ValueError(f"Unsupported serialization format for path: {path}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_json_string(self, data: Any, indent: int = 2) -> str:
        """Serialise *data* to a JSON string without writing to disk."""
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def from_json_string(self, text: str) -> Any:
        """Deserialise *text* from a JSON string."""
        return json.loads(text)

    def iter_jsonl(self, path: Union[str, Path]) -> Iterator[Any]:
        """Iterate over a JSON-Lines file, yielding one object per line."""
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def write_jsonl(self, path: Union[str, Path], records: List[Any]) -> Path:
        """Write *records* to a JSON-Lines file (one JSON object per line)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return p


# ---------------------------------------------------------------------------
# Rate limiter (used by AIClient)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter.

    Example::

        limiter = RateLimiter(max_calls=10, period=60)
        for item in items:
            limiter.wait()  # blocks if necessary
            process(item)
    """

    def __init__(self, max_calls: int = 60, period: float = 60.0):
        """
        Args:
            max_calls: Maximum allowed calls within *period* seconds.
            period:    Time window in seconds.
        """
        self.max_calls = max_calls
        self.period = period
        self._timestamps: List[float] = []

    def wait(self):
        """Block until a call token is available."""
        now = time.monotonic()
        # Purge timestamps outside the current window
        self._timestamps = [t for t in self._timestamps if now - t < self.period]

        if len(self._timestamps) >= self.max_calls:
            sleep_time = self.period - (now - self._timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._timestamps = self._timestamps[1:]

        self._timestamps.append(time.monotonic())

    def is_allowed(self) -> bool:
        """Return True if a call is immediately allowed (non-blocking check)."""
        now = time.monotonic()
        active = [t for t in self._timestamps if now - t < self.period]
        return len(active) < self.max_calls
