from typing import Optional, Dict, Tuple

import logging
import sys


class VerbosePrinter:
    """Conditional logger controlled by a global verbosity.

    Attributes:
        verbose (int): Global verbosity. 0 disables all output, 1 or higher enables it.
    """

    _COLORS = {
        "black": "\x1b[30m",
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "white": "\x1b[37m",
        "orange": "\x1b[38;5;208m",  # 256-color extension
        "reset": "\x1b[0m",
    }

    _STYLES = {
        "bold": "\x1b[1m",
        "italic": "\x1b[3m",
    }

    _NAME_TO_LEVEL = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, verbose: int = 1, logger: Optional[logging.Logger] = None):
        """Initialize the logger.

        Args:
            verbose (int): Initial verbosity, non-negative. Default is 1.
            logger (Optional[logging.Logger]): External logger to use. If None, a simple
                stdout logger is created.

        Raises:
            TypeError: If verbose is not an int.
            ValueError: If verbose < 0.
        """
        self._verbose = 1
        self.verbose = verbose

        self._logger = logger or logging.getLogger(f"VerbosePrinter.{id(self)}")
        if logger is None:
            self._logger.setLevel(logging.DEBUG)
            self._logger.propagate = False
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.handlers.clear()
            self._logger.addHandler(handler)

    # ---------- helpers ----------

    @staticmethod
    def _require_int_at_least(name: str, value: object, minimum: int) -> int:
        """Validate that value is an int and >= minimum.

        Args:
            name (str): Parameter name.
            value (object): Value to validate.
            minimum (int): Inclusive minimum.

        Returns:
            int: Validated integer.

        Raises:
            TypeError: If value is not an int.
            ValueError: If value < minimum.
        """
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    @staticmethod
    def _require_bool(name: str, value: object) -> bool:
        """Validate that value is a bool.

        Args:
            name (str): Parameter name.
            value (object): Value to validate.

        Returns:
            bool: Validated boolean.

        Raises:
            TypeError: If value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a bool")
        return value

    def _validate_color_name(self, color: Optional[str]) -> Optional[str]:
        """Validate color name against the supported palette.

        Args:
            color (Optional[str]): Color name or None.

        Returns:
            Optional[str]: The same color name or None.

        Raises:
            TypeError: If color is not a str or None.
            ValueError: If color is unknown.
        """
        if color is None:
            return None
        if not isinstance(color, str):
            raise TypeError("color must be a str or None")
        if color.lower() not in self._COLORS:
            raise ValueError(
                "unknown color, supported: black, red, green, yellow, blue, orange, magenta, cyan, white, reset"
            )
        return color.lower()

    def _normalize_style_param(self, style: Optional[Dict[str, bool]]) -> Tuple[bool, bool]:
        """Normalize style dict to booleans.

        Args:
            style (Optional[Dict[str, bool]]): Dict with keys 'bold' and/or 'italic'.

        Returns:
            Tuple[bool, bool]: (bold, italic) flags.

        Raises:
            TypeError: If style is not a dict or None, or values are not bools.
        """
        if style is None:
            return False, False
        if not isinstance(style, dict):
            raise TypeError("style must be a dict or None")
        bold = self._require_bool("style['bold']", bool(style.get("bold", False)))
        italic = self._require_bool("style['italic']", bool(style.get("italic", False)))
        return bold, italic

    def _coerce_log_level(self, log_level: object) -> int:
        """Coerce textual or numeric log level to a logging level number.

        Args:
            log_level (object): String like "INFO" or an integer level.

        Returns:
            int: A valid logging level number.

        Raises:
            TypeError: If log_level is neither str nor int.
            ValueError: If log_level is an int < 1.
        """
        if isinstance(log_level, int):
            return self._require_int_at_least("log_level", log_level, 1)
        if isinstance(log_level, str):
            level = self._NAME_TO_LEVEL.get(log_level.upper())
            return level if level is not None else logging.INFO
        raise TypeError("log_level must be str or int")

    def _should_emit(self, level: int) -> bool:
        """Check verbosity gate.

        Args:
            level (int): Minimum verbosity required, must be >= 1.

        Returns:
            bool: True if the message should be emitted.
        """
        self._require_int_at_least("level", level, 1)
        return self._verbose > 0 and self._verbose >= level

    def _primary_handler(self) -> logging.Handler:
        """Return the first handler, creating a default one if absent.

        Returns:
            logging.Handler: Primary handler.
        """
        if not self._logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(h)
        return self._logger.handlers[0]

    # ---------- properties ----------

    @property
    def verbose(self) -> int:
        """Get current verbosity.

        Returns:
            int: 0 disables output, 1 or higher enables it.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: int) -> None:
        """Set the global verbosity.

        Args:
            value (int): Non-negative integer.

        Raises:
            TypeError: If value is not an int.
            ValueError: If value < 0.
        """
        self._verbose = self._require_int_at_least("verbose", value, 0)

    # ---------- tag preset generator ----------

    def gen_tag(self, name: Optional[str] = None, type: str = "simple") -> str:
        """Generate a logging format string preset.

        Supported types:
            - "simple": "[<NAME> %(levelname)s] %(message)s"
            - "fileline": "[<NAME> %(filename)s:%(lineno)d %(levelname)s] %(message)s"
            - "time": "[%(asctime)s %(levelname)s] %(message)s"

        Args:
            name (Optional[str]): Label to replace "ARARAS". If None, defaults to "ARARAS"
                for types "simple" and "fileline". Ignored for "time".
            type (str): One of "simple", "fileline", "time". Case-insensitive.

        Returns:
            str: A logging format string suitable for logging.Formatter.

        Raises:
            ValueError: If type is unknown.
        """
        kind = type.lower()
        label = name if name is not None else "ARARAS"
        if kind == "simple":
            return f"[{label} %(levelname)s] %(message)s"
        if kind == "fileline":
            return f"[{label} %(filename)s:%(lineno)d %(levelname)s] %(message)s"
        if kind == "time":
            return "[%(asctime)s %(levelname)s] %(message)s"
        raise ValueError('type must be one of "simple", "fileline", "time"')

    # ---------- styling primitives ----------

    def color(self, message: object, color: str, *, add_reset: bool = True) -> str:
        """Wrap a message with an ANSI color code.

        Args:
            message (object): Any object, converted with str().
            color (str): One of black, red, green, yellow, blue, orange, magenta, cyan, white, reset.
            add_reset (bool): If True, append a reset code at the end. Default is True.

        Returns:
            str: Colored message.

        Raises:
            ValueError: If color is unknown.
            TypeError: If add_reset is not a bool.
        """
        self._require_bool("add_reset", add_reset)
        cname = self._validate_color_name(color)
        code = self._COLORS[cname]
        tail = self._COLORS["reset"] if add_reset else ""
        return f"{code}{str(message)}{tail}"

    def style(
        self, message: object, *, bold: bool = False, italic: bool = False, add_reset: bool = True
    ) -> str:
        """Wrap a message with ANSI style codes.

        Args:
            message (object): Any object, converted with str().
            bold (bool): If True, apply bold.
            italic (bool): If True, apply italic.
            add_reset (bool): If True, append a reset code at the end. Default is True.

        Returns:
            str: Styled message.

        Raises:
            TypeError: If bold, italic, or add_reset is not a bool.
        """
        self._require_bool("bold", bold)
        self._require_bool("italic", italic)
        self._require_bool("add_reset", add_reset)
        if not bold and not italic:
            return str(message)
        prefix = ""
        if bold:
            prefix += self._STYLES["bold"]
        if italic:
            prefix += self._STYLES["italic"]
        tail = self._COLORS["reset"] if add_reset else ""
        return f"{prefix}{str(message)}{tail}"

    # ---------- API ----------

    def printf(
        self,
        message: str,
        level: int = 1,
        tag: object = "",
        color: Optional[str] = None,
        style: Optional[Dict[str, bool]] = None,
        end: str = "\n",
    ) -> None:
        """Print a message if verbosity is high enough.

        The `color` and `style` parameters affect the entire output, including the tag.

        Args:
            message (str): Text to print.
            level (int): Minimum verbosity required, must be >= 1. Default is 1.
            tag (object): Prefix placed before the message, converted with str().
            color (Optional[str]): Text color. Supported values are
                {"black","red","green","yellow","blue","orange","magenta","cyan","white","reset"}.
                Default is None for no color.
            style (Optional[Dict[str, bool]]): Style flags, for example
                {"bold": True, "italic": False}. Default is None for no style.
            end (str): Line terminator for print. Default is newline.

        Returns:
            None

        Raises:
            TypeError: If level is not an int, color is invalid type, or style is invalid type.
            ValueError: If level < 1, or color is unknown.
        """
        if not self._should_emit(level):
            return

        color_name = self._validate_color_name(color)
        bold, italic = self._normalize_style_param(style)

        payload = f"{str(tag)}{message}"
        if bold or italic:
            payload = self.style(payload, bold=bold, italic=italic, add_reset=(color_name is None))
        if color_name is not None:
            payload = self.color(payload, color_name, add_reset=True)

        print(payload, end=end)

    def logf(
        self,
        message: str,
        level: int = 1,
        log_level: object = "INFO",
        tag: object = "",
        color: Optional[str] = None,
        style: Optional[Dict[str, bool]] = None,
    ) -> None:
        """Log a message with an explicit log level if verbosity is high enough.

        The `tag` parameter supports two modes:
            1) Prefix mode, any non-formatter string is prepended to the message.
            2) Formatter mode, if `tag` contains logging placeholders like
               "%(message)s" or "%(levelname)s", it is treated as a full format
               string and applied only for this call.
               
        Use the gen_tag() method to generate common formatter presets.

        The `color` and `style` parameters affect the entire output, including the tag
        and any formatter-produced fields.

        Color options:
            {"black","red","green","yellow","blue","orange","magenta","cyan","white","reset"}

        Style options:
            {"bold": bool, "italic": bool}

        Args:
            message (str): Text to log.
            level (int): Minimum verbosity required, must be >= 1. Default is 1.
            log_level (object): String level like "INFO", "WARNING", "ERROR",
                or an integer logging level. Unknown strings default to INFO.
            tag (object): Prefix string or a format string returned by `gen_tag`.
            color (Optional[str]): Color name from the supported set, or None.
            style (Optional[Dict[str, bool]]): Dict with keys 'bold' and/or 'italic', or None.

        Returns:
            None

        Raises:
            TypeError: If level is not an int, color/style have invalid types, or log_level has invalid type.
            ValueError: If level < 1, color is unknown, or log_level is an int < 1.
        """
        if not self._should_emit(level):
            return

        lvlno = self._coerce_log_level(log_level)
        color_name = self._validate_color_name(color)
        bold, italic = self._normalize_style_param(style)

        tag_str = str(tag)
        is_formatter = any(tok in tag_str for tok in ("%(", "%(message)s", "%(levelname)s", "%(asctime)s"))

        if is_formatter and "%(message)s" in tag_str:
            handler = self._primary_handler()
            prev = handler.formatter

            class _OneShotFormatter(logging.Formatter):
                def __init__(self, fmt: str, datefmt: Optional[str], apply):
                    super().__init__(fmt=fmt, datefmt=datefmt)
                    self._apply = apply

                def format(self, record: logging.LogRecord) -> str:
                    s = super().format(record)
                    return self._apply(s)

            def apply_all(text: str) -> str:
                out = text
                if bold or italic:
                    out = self.style(out, bold=bold, italic=italic, add_reset=(color_name is None))
                if color_name is not None:
                    out = self.color(out, color_name, add_reset=True)
                return out

            datefmt = "%Y-%m-%d %H:%M:%S" if "%(asctime)s" in tag_str else None
            try:
                handler.setFormatter(_OneShotFormatter(tag_str, datefmt, apply_all))
                self._logger.log(lvlno, message)
            finally:
                handler.setFormatter(prev)
        else:
            payload = f"{tag_str}{message}"
            if bold or italic:
                payload = self.style(payload, bold=bold, italic=italic, add_reset=(color_name is None))
            if color_name is not None:
                payload = self.color(payload, color_name, add_reset=True)
            self._logger.log(lvlno, payload)


# # Example usage
# if __name__ == "__main__":
#     vp = VerbosePrinter()

#     vp.printf("hello", tag="[ARARAS] ", color="cyan", style={"bold": True})
#     vp.logf("simple preset", tag=vp.gen_tag(), color="yellow")
#     vp.logf(
#         "timestamp preset", log_level="ERROR", tag=vp.gen_tag(type="time"), color="red", style={"bold": True}
#     )
#     vp.verbose = 0
#     vp.logf("silenced", log_level="INFO")
