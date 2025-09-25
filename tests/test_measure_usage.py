"""Demonstrate ``measure_callable_resource_usage`` on a simple workload.

Run this module directly to see the collected metrics:

    python tests/demo_measure_usage.py
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("optuna", types.ModuleType("optuna"))
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

from araras.utils.system import measure_callable_resource_usage


def _allocate_blocks(block_size: int, iterations: int) -> int:
    """Allocate and release bytearrays to create a predictable RAM profile."""

    blocks = []
    for _ in range(iterations):
        blocks.append(bytearray(block_size))
        time.sleep(0.05)
    blocks.clear()
    return block_size * iterations


def main() -> None:
    block_size = 256 * 1024  # 256 KiB per allocation
    iterations = 8
    expected_bytes = block_size * iterations

    summary, result = measure_callable_resource_usage(
        _allocate_blocks,
        block_size,
        iterations,
        metrics=("ram",),
    )

    system_ram = summary.get("system_ram")

    print(f"Expected peak allocation: {expected_bytes} bytes")
    print(f"Callable reported allocating {result} bytes")
    if isinstance(system_ram, dict):
        print("Measured RAM usage (bytes):")
        for key in ("before", "current", "difference", "final"):
            value = system_ram.get(key, "n/a")
            print(f"  {key}: {value}")
    else:
        print(f"RAM metrics: {system_ram}")


if __name__ == "__main__":
    main()
