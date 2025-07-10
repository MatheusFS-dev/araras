from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ————————————————————————————— File Type Handler ———————————————————————————— #
class FileTypeHandler:
    """File type detection and command generation with caching."""

    # Class-level cache for performance optimization (bounded to prevent memory growth)
    _file_type_cache: Dict[str, str] = {}
    _command_cache: Dict[str, List[str]] = {}
    _CACHE_LIMIT = 100

    @classmethod
    def get_file_type(cls, file_path: Path) -> str:
        """Determine file type with O(1) cached lookup.

        Args:
            file_path: Path to the file

        Returns:
            File type ('python', 'notebook', 'unknown')
        """
        path_str = str(file_path)

        # O(1) cache lookup for performance
        if path_str in cls._file_type_cache:
            return cls._file_type_cache[path_str]

        # Single suffix check (most efficient approach)
        suffix = file_path.suffix.lower()
        if suffix == ".py":
            file_type = "python"
        elif suffix == ".ipynb":
            file_type = "notebook"
        else:
            file_type = "unknown"

        # Bounded cache to prevent memory growth
        if len(cls._file_type_cache) < cls._CACHE_LIMIT:
            cls._file_type_cache[path_str] = file_type

        return file_type

    @classmethod
    def build_execution_command(cls, file_path: Path, success_flag_file: str) -> Tuple[List[str], str]:
        """Build optimized execution command based on file type.

        Args:
            file_path: Path to file to execute
            success_flag_file: Path to success flag file

        Returns:
            Tuple of (command_list, execution_type)

        Raises:
            ValueError: If file type is unsupported
        """
        path_str = str(file_path.resolve())
        cache_key = f"{path_str}:{success_flag_file}"

        # Check command cache for performance optimization
        if cache_key in cls._command_cache:
            cached_cmd = cls._command_cache[cache_key]
            return cached_cmd.copy(), cls.get_file_type(file_path)

        file_type = cls.get_file_type(file_path)

        if file_type == "python":
            # Direct Python execution (most efficient)
            command = [sys.executable, "-u", str(file_path), success_flag_file]

        elif file_type == "notebook":
            # This should never be reached after conversion, but keeping for safety
            raise ValueError(f"Notebook files should be converted to Python first: {file_path}")

        else:
            raise ValueError(f"Unsupported file type: {file_type} for {file_path}")

        # Cache command with bounded size to prevent memory growth
        if len(cls._command_cache) < cls._CACHE_LIMIT:
            cls._command_cache[cache_key] = command.copy()

        return command, file_type

    @classmethod
    def validate_file(cls, file_path: str) -> Path:
        """Validate file existence and type with early exit pattern.

        Args:
            file_path: Path string to validate

        Returns:
            Resolved Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        path_obj = Path(file_path)

        # Early exit validation for performance
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        file_type = cls.get_file_type(path_obj)
        if file_type == "unknown":
            raise ValueError(f"Unsupported file type: {path_obj.suffix}. Supported: .py, .ipynb")

        return path_obj.resolve()



