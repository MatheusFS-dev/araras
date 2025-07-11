"""
Utility class for detecting file types and building execution commands.

Classes:
    - FileTypeHandler: Provides methods to identify file types and construct
      commands for running them.

Example:
    >>> from araras.kernel.file_type_handler import FileTypeHandler
    >>> FileTypeHandler.build_execution_command(Path("train.py"), "success.txt")
"""
from araras.commons import *

import sys
from pathlib import Path


class FileTypeHandler:
    """File type detection and command generation with caching."""

    _file_type_cache: Dict[str, str] = {}
    _command_cache: Dict[str, List[str]] = {}
    _CACHE_LIMIT = 100

    @classmethod
    def get_file_type(cls, file_path: Path) -> str:
        """Return the file type for a path."""
        path_str = str(file_path)
        if path_str in cls._file_type_cache:
            return cls._file_type_cache[path_str]

        suffix = file_path.suffix.lower()
        if suffix == ".py":
            file_type = "python"
        elif suffix == ".ipynb":
            file_type = "notebook"
        else:
            file_type = "unknown"

        if len(cls._file_type_cache) < cls._CACHE_LIMIT:
            cls._file_type_cache[path_str] = file_type

        return file_type

    @classmethod
    def build_execution_command(cls, file_path: Path, success_flag_file: str) -> Tuple[List[str], str]:
        """Build execution command based on file type."""
        path_str = str(file_path.resolve())
        cache_key = f"{path_str}:{success_flag_file}"
        if cache_key in cls._command_cache:
            cached_cmd = cls._command_cache[cache_key]
            return cached_cmd.copy(), cls.get_file_type(file_path)

        file_type = cls.get_file_type(file_path)

        if file_type == "python":
            command = [sys.executable, "-u", str(file_path), success_flag_file]
        elif file_type == "notebook":
            raise ValueError(f"Notebook files should be converted to Python first: {file_path}")
        else:
            raise ValueError(f"Unsupported file type: {file_type} for {file_path}")

        if len(cls._command_cache) < cls._CACHE_LIMIT:
            cls._command_cache[cache_key] = command.copy()

        return command, file_type

    @classmethod
    def validate_file(cls, file_path: str) -> Path:
        """Validate file existence and type."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        file_type = cls.get_file_type(path_obj)
        if file_type == "unknown":
            raise ValueError(f"Unsupported file type: {path_obj.suffix}. Supported: .py, .ipynb")

        return path_obj.resolve()
