"""Real-time visualization reporter using matplotlib."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt


@dataclass
class _ReporterState:
    fig: plt.Figure
    axes: Dict[str, plt.Axes]
    lines: Dict[str, plt.Line2D]
    history: Dict[str, List[tuple[float, float]]]
    summary_text: plt.Text


_state: Optional[_ReporterState] = None


def report(
    y_data: Dict[str, float],
    x_data: Dict[str, float],
    summary_data: Dict[str, Any],
    *,
    title: str = "Araras Reporter",
) -> None:
    """Update plots and summary information in real time.

    Initializes the interface on the first call and updates the stored
    history and figure elements on subsequent calls.

    Args:
        y_data: Mapping of series name to latest Y value.
        x_data: Mapping of series name to latest X value.
        summary_data: Mapping of summary label to value displayed as text.
        title: Window title for the matplotlib figure.
    """
    global _state

    if _state is None:
        plt.ion()
        n_plots = len(y_data)
        fig, axes = plt.subplots(n_plots, 1, sharex=False)
        if n_plots == 1:
            axes = [axes]
        fig.suptitle(title)

        summary_text = fig.text(0.5, 0.01, "", ha="center", va="bottom")

        _state = _ReporterState(fig, {}, {}, {}, summary_text)

        for ax, name in zip(axes, y_data.keys()):
            line, = ax.plot([], [], marker="o")
            ax.set_title(name)
            _state.axes[name] = ax
            _state.lines[name] = line
            _state.history[name] = []
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Update data and plots
    for name, y in y_data.items():
        x = x_data.get(name, len(_state.history.get(name, [])))
        hist = _state.history.setdefault(name, [])
        hist.append((x, y))
        xs, ys = zip(*hist)
        line = _state.lines.get(name)
        if line is None:
            # Create new subplot if a new series appears after initialization
            ax = _state.fig.add_subplot(len(_state.axes) + 1, 1, len(_state.axes) + 1)
            line, = ax.plot([], [], marker="o")
            ax.set_title(name)
            _state.axes[name] = ax
            _state.lines[name] = line
            _state.fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        line.set_data(xs, ys)
        ax = _state.axes[name]
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(min(xs), max(xs))

    # Update summary text
    summary_str = "\n".join(f"{k}: {v}" for k, v in summary_data.items())
    _state.summary_text.set_text(summary_str)

    _state.fig.canvas.draw()
    _state.fig.canvas.flush_events()

