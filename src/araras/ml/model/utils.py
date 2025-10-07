from araras.core import *

from typing import Literal, Optional, Tuple


DeviceKind = Literal["cpu", "gpu", "both"]


def parse_device_spec(device: str) -> Tuple[DeviceKind, Optional[int]]:
    """Parse a canonical device specification string.

    The helper enforces the ``"device/index"`` convention adopted across the
    project to describe profiling targets. GPU-only profiling is expressed as
    ``"gpu/<index>"``, combined CPU and GPU profiling uses ``"both/<index>"``
    to indicate which GPU pairs with the CPU run, and CPU-only profiling is
    denoted simply as ``"cpu"``.

    Args:
        device (str): Canonical device string. Must be ``"cpu"`` or follow the
            ``"<device>/<index>"`` pattern where ``<device>`` is ``"gpu"`` or
            ``"both"`` and ``<index>`` is a non-negative integer.

    Returns:
        Tuple[DeviceKind, Optional[int]]: Normalised device kind coupled with the
        GPU index when applicable. CPU-only requests return ``None`` for the
        index component.

    Raises:
        TypeError: If ``device`` is not provided as a string.
        ValueError: If the specification is empty, omits the index for GPU or
            combined profiling, or uses an unsupported device kind.

    Notes:
        The parser guarantees non-negative GPU indices but does not verify that
        the referenced GPU exists on the running system.

    Warnings:
        Passing ``"gpu/-1"`` or similar negative indices raises ``ValueError``.
        Callers must clamp or validate user input before invoking this helper
        when working with untrusted sources.
    """

    if not isinstance(device, str):
        raise TypeError("device must be a string in the form 'cpu' or 'device/index'")

    normalized = device.strip().lower()
    if not normalized:
        raise ValueError("device cannot be empty")

    if normalized == "cpu":
        return "cpu", None

    if "/" not in normalized:
        raise ValueError(
            "device must follow the 'device/index' format for GPU or combined profiling"
        )

    kind_text, index_text = (part.strip() for part in normalized.split("/", 1))
    if kind_text not in {"gpu", "both"}:
        raise ValueError(f"Unsupported device kind '{kind_text}'. Use 'gpu' or 'both'.")
    if not index_text:
        raise ValueError("device index must be provided for GPU or combined profiling")

    try:
        index = int(index_text)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError("device index must be an integer") from exc

    if index < 0:
        raise ValueError("device index must be non-negative")

    device_kind: DeviceKind = "gpu" if kind_text == "gpu" else "both"
    return device_kind, index


def capture_model_summary(model):
    """Capture model summary as a string.

    Args:
        model (Any): Keras model

    Returns:
        str: Model summary as string
    """
    summary_list = []
    model.summary(
        print_fn=lambda x: summary_list.append(x),
        expand_nested=True,
        show_trainable=True,
    )
    return "\n".join(summary_list)
