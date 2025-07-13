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

    Args:
        iterable (Iterable): The iterable to wrap and iterate over.
        description (str): A description to display alongside the progress bar.
        total (int): The total number of iterations.

    Yields:
        Any: Items from the provided iterable, while displaying the progress bar.
    """
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="white",
            finished_style="white", style="black"),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )

    with progress:
        yield from progress.track(iterable, total=total, description=description)
