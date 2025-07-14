from araras.core import *

import matplotlib.pyplot as plt
import optuna
import numpy as np
import scipy.interpolate

from araras.ml.optuna.analyzer import (
    PLOT_CFG,
    draw_warning_box,
    save_plot,
)


def plot_edf(study: optuna.Study, dirs: Dict[str, str], create_plotly: bool = False) -> None:
    """Plot the empirical distribution function of objective values.

    Parameters
    ----------
    create_plotly : bool
        Whether to save an interactive HTML version of the plot.
    """
    df = study.trials_dataframe()
    df = df.query("state == 'COMPLETE'")
    values = df["value"].astype(float).sort_values()
    if values.empty:
        fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
        draw_warning_box(ax, "No completed trials for EDF plot.")
        ax.set_title(
            "Empirical Distribution of Study Values",
            pad=PLOT_CFG.title_pad,
            fontsize=PLOT_CFG.standalone_title_fs,
        )
        plt.tight_layout()
        save_plot(fig, dirs, "study_edf", "figs", create_plotly)
        plt.close(fig)
        return

    ecdf = np.arange(1, len(values) + 1) / len(values)

    # Interpolate values and ecdf for higher resolution
    interp_func = scipy.interpolate.interp1d(values, ecdf, kind="linear")
    high_res_values = np.linspace(values.min(), values.max(), num=1000)  # Increase resolution
    high_res_ecdf = interp_func(high_res_values)

    fig, ax = plt.subplots(figsize=PLOT_CFG.standalone_size)
    ax.step(high_res_values, high_res_ecdf, where="post", color="black")
    ax.set_xlabel(PLOT_CFG.study_value_label, fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_ylabel("Cumulative Proportion", fontsize=PLOT_CFG.standalone_label_fs)
    ax.set_title(
        "Empirical Distribution of Study Values",
        pad=PLOT_CFG.title_pad,
        fontsize=PLOT_CFG.standalone_title_fs,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pfig = None
    if create_plotly:
        import plotly.graph_objects as go

        pfig = go.Figure(go.Scatter(x=high_res_values, y=high_res_ecdf, mode="lines"))
        pfig.update_layout(
            xaxis_title=PLOT_CFG.study_value_label,
            yaxis_title="Cumulative Proportion",
            title="Empirical Distribution of Study Values",
            template="plotly_white",
        )

    save_plot(fig, dirs, "study_edf", "figs", create_plotly, pfig)
    plt.close(fig)
