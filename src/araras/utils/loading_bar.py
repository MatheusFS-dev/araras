from typing import Iterable, Any
from tqdm import tqdm

# ANSI escape codes
RESET = "\x1b[0m"
ANSI = {
    "white": "\x1b[37m",
    "yellow": "\x1b[33m",
    "red": "\x1b[31m",
    "orange": "\x1b[38;5;208m",
    "blue": "\x1b[94m",
    "green": "\x1b[32m",
    "cyan": "\x1b[36m",
    "magenta": "\x1b[35m",
}


def gen_loading_bar(
    iterable: Iterable[Any],
    *,
    description: str,
    total: int,
    bar_color: str = "white",
):
    """Iterate with a progress bar whose bar and counters are colored.

    Only the bar visual, the percentage, and the n/total counters receive color.
    The description and other texts remain uncolored.

    Args:
        iterable (Iterable[Any]): Items to iterate over.
        description (str): Text shown before the bar.
        total (int): Expected number of iterations, must be >= 1.
        bar_color (str): Color name for bar and counters only. Accepted values:
            white, yellow, red, orange, blue, green, cyan, magenta. Default is "white".

    Yields:
        Any: Items from ``iterable``.

    Raises:
        TypeError: If ``description`` is not str or ``total`` is not int.
        ValueError: If ``total`` < 1 or ``bar_color`` is unknown.
    """
    if not isinstance(description, str):
        raise TypeError("description must be a str")
    if not isinstance(total, int):
        raise TypeError("total must be an int")
    if total < 1:
        raise ValueError("total must be >= 1")

    color = ANSI.get(bar_color.lower())
    if color is None:
        raise ValueError(f"unknown bar_color '{bar_color}'")

    # Color only {percentage}, {bar}, and {n_fmt}/{total_fmt}
    bar_fmt = (
        "{desc}: "
        + f"{color}"
        + "{percentage:3.0f}%"
        + " ["
        + "{bar}"
        + "] "
        + "{n_fmt}/{total_fmt}"
        + f"{RESET}"
        + " in {remaining}"
    )

    with tqdm(
        iterable,
        total=total,
        desc=description,
        ncols=70,
        ascii=".#",
        bar_format=bar_fmt,
        unit="",  # remove default "it" label
    ) as pbar:
        try:
            for item in pbar:
                yield item
        finally:
            if pbar.total is not None:
                remaining = pbar.total - pbar.n
                if remaining > 0:
                    pbar.update(remaining)
                elif remaining < 0:
                    pbar.n = pbar.total
            pbar.refresh()
            pbar.close()
