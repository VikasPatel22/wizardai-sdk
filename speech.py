"""
WizardAI Plugin System
-----------------------
Base class and manager for extending WizardAI with custom skills / plugins.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .exceptions import PluginError
from .utils import Logger


# ---------------------------------------------------------------------------
# PluginBase
# ---------------------------------------------------------------------------

class PluginBase(ABC):
    """Abstract base class for all WizardAI plugins.

    Subclass this and implement :meth:`on_message` to create a custom skill.

    Example::

        class WeatherPlugin(PluginBase):
            name = "weather"
            description = "Provides current weather for a given city."
            version = "1.0.0"
            triggers = ["weather in *", "what's the weather in *"]

            def setup(self):
                self.api_key = self.config.get("api_key", "")

            def on_message(self, text: str, context: dict) -> Optional[str]:
                city = text.split("in", 1)[-1].strip()
                return f"The weather in {city} is sunny, 25°C."

        manager = PluginManager()
        manager.register(WeatherPlugin)
    """

    # ------------------------------------------------------------------
    # Class-level metadata (override in subclass)
    # ------------------------------------------------------------------

    name: str = "unnamed_plugin"
    description: str = ""
    version: str = "0.0.1"
    author: str = ""
    triggers: List[str] = []   # Pattern strings this plugin handles

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            config: Plugin-specific configuration dictionary.
            logger: Optional Logger instance.
        """
        self.config = config or {}
        self.logger = logger or Logger(f"Plugin:{self.name}")
        self._enabled = True
        self.setup()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self):
        """Called once after __init__.  Override to initialise resources."""
        pass

    def teardown(self):
        """Called when the plugin is unregistered.  Override to clean up."""
        pass

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_message(self, text: str, context: Dict[str, Any]) -> Optional[str]:
        """Process *text* and return a response, or None to pass through.

        Args:
            text:    The user's input (already matched against a trigger).
            context: Shared context dict from the WizardAI core.

        Returns:
            A response string, or None if this plugin cannot handle the input.
        """

    def on_start(self):
        """Called when the WizardAI session starts.  Override as needed."""
        pass

    def on_stop(self):
        """Called when the WizardAI session ends.  Override as needed."""
        pass

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable(self):
        """Enable this plugin."""
        self._enabled = True
        self.logger.info(f"Plugin '{self.name}' enabled.")

    def disable(self):
        """Disable this plugin (on_message will not be called)."""
        self._enabled = False
        self.logger.info(f"Plugin '{self.name}' disabled.")

    @property
    def is_enabled(self) -> bool:
        """True if the plugin is currently enabled."""
        return self._enabled

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        status = "enabled" if self._enabled else "disabled"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Manages lifecycle, registration, and dispatch of WizardAI plugins.

    Example::

        manager = PluginManager()
        manager.register(WeatherPlugin, config={"api_key": "..."})
        manager.register(JokePlugin)

        # Dispatch user input to matching plugins
        response = manager.dispatch("weather in Paris", context={})
        print(response)  # "The weather in Paris is sunny, 25°C."

        # List all active plugins
        for plugin in manager.list_plugins():
            print(plugin)
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger("PluginManager")
        self._plugins: Dict[str, PluginBase] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        plugin_cls: Type[PluginBase],
        config: Optional[Dict[str, Any]] = None,
        name_override: Optional[str] = None,
    ) -> PluginBase:
        """Instantiate and register a plugin class.

        Args:
            plugin_cls:    Subclass of :class:`PluginBase` to register.
            config:        Configuration dict passed to the plugin.
            name_override: Use this name instead of ``plugin_cls.name``.

        Returns:
            The instantiated plugin object.

        Raises:
            PluginError: If a plugin with the same name is already registered.
        """
        if not (inspect.isclass(plugin_cls) and issubclass(plugin_cls, PluginBase)):
            raise PluginError(
                f"{plugin_cls!r} is not a PluginBase subclass.",
                plugin_name=str(plugin_cls),
            )

        name = name_override or plugin_cls.name
        if name in self._plugins:
            raise PluginError(
                f"A plugin named '{name}' is already registered. "
                "Use name_override to use a different name.",
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
        """Unregister and teardown a plugin by name.

        Returns:
            True if the plugin was found and removed, False otherwise.
        """
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
        """Return the plugin instance with the given name, or None."""
        return self._plugins.get(name)

    def list_plugins(self, enabled_only: bool = False) -> List[PluginBase]:
        """Return a list of registered plugin instances.

        Args:
            enabled_only: If True, only return enabled plugins.
        """
        plugins = list(self._plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p.is_enabled]
        return plugins

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Route *text* to the first enabled plugin that returns a response.

        Iterates through registered plugins in insertion order. Returns the
        first non-None response, or None if no plugin handles the input.

        Args:
            text:    User input string.
            context: Shared context dict (mutated by plugins as needed).

        Returns:
            Response string from the handling plugin, or None.
        """
        ctx = context or {}
        for plugin in self._plugins.values():
            if not plugin.is_enabled:
                continue
            try:
                result = plugin.on_message(text, ctx)
                if result is not None:
                    self.logger.debug(
                        f"[Dispatch] Plugin '{plugin.name}' handled: {text[:40]!r}"
                    )
                    return result
            except Exception as exc:
                self.logger.error(
                    f"[Dispatch] Plugin '{plugin.name}' raised an error: {exc}"
                )
        return None

    def dispatch_all(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str]]:
        """Route *text* to all enabled plugins and collect responses.

        Unlike :meth:`dispatch`, this does not stop at the first match.

        Returns:
            List of ``(plugin_name, response)`` tuples for all plugins that
            returned a non-None response.
        """
        from typing import Tuple  # local import to avoid circular
        ctx = context or {}
        results = []
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
    # Dynamic loading
    # ------------------------------------------------------------------

    def load_from_file(
        self,
        path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> PluginBase:
        """Dynamically load and register the first PluginBase subclass found
        in a Python source file.

        Args:
            path:   Path to the Python file containing the plugin class.
            config: Configuration dict for the plugin.

        Returns:
            The registered plugin instance.

        Raises:
            PluginError: If no valid plugin class is found in the file.
        """
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
            spec.loader.exec_module(module)
        except Exception as exc:
            raise PluginError(f"Error executing plugin file {p}: {exc}") from exc

        # Find PluginBase subclass
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
            "Make sure your plugin class inherits from PluginBase."
        )

    def load_from_directory(
        self,
        directory: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[PluginBase]:
        """Load all plugins from ``*.py`` files in *directory*.

        Returns:
            List of successfully registered plugin instances.
        """
        d = Path(directory).resolve()
        if not d.is_dir():
            raise PluginError(f"Not a directory: {d}")

        loaded = []
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                plugin = self.load_from_file(py_file, config=config)
                loaded.append(plugin)
            except PluginError as exc:
                self.logger.warning(f"Skipping {py_file.name}: {exc}")

        self.logger.info(
            f"Loaded {len(loaded)} plugin(s) from {d}"
        )
        return loaded

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_all(self):
        """Call ``on_start()`` on all enabled plugins."""
        for plugin in self._plugins.values():
            if plugin.is_enabled:
                try:
                    plugin.on_start()
                except Exception as exc:
                    self.logger.error(f"Plugin start error ({plugin.name}): {exc}")

    def stop_all(self):
        """Call ``on_stop()`` on all enabled plugins."""
        for plugin in self._plugins.values():
            try:
                plugin.on_stop()
            except Exception as exc:
                self.logger.error(f"Plugin stop error ({plugin.name}): {exc}")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._plugins)

    def __repr__(self):
        names = list(self._plugins.keys())
        return f"PluginManager(plugins={names})"


# ---------------------------------------------------------------------------
# Typing fix (Tuple used in dispatch_all)
# ---------------------------------------------------------------------------
from typing import Tuple  # noqa: E402 – needed after class definition
