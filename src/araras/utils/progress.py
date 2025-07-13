from __future__ import annotations

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


def white_track(iterable, *, description: str, total: int):
    """Iterate with a white progress bar showing ``done/total``.

    Parameters
    ----------
    iterable:
        Iterable to wrap.
    description:
        Progress bar description.
    total:
        Number of iterations.

    Yields
    ------
    Items from ``iterable`` with progress display.
    """
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, style="white"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )

    with progress:
        yield from progress.track(iterable, total=total, description=description)
