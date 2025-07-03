import os
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances

from .analyze import PLOT_CFG, save_data_for_latex

def plot_param_importances(study: optuna.Study, dirs: Dict[str, str]) -> None:
    """
    Generate and save parameter importance analysis.

    This function computes parameter importances using Optuna's built-in
    importance calculation and creates both a CSV table and bar chart
    visualization to identify which parameters most influence the objective.

    Args:
        study (optuna.Study): Optuna study object containing optimization history
        dirs (Dict[str, str]): Directory paths for saving outputs

    Returns:
        None: Saves importance table as CSV and bar chart as pdf
    """
    # Calculate parameter importances using Optuna's algorithm
    importances = get_param_importances(study)

    # Convert to DataFrame and sort by importance (descending)
    df_imp = pd.DataFrame(list(importances.items()), columns=["Parameter", "Importance"]).sort_values(
        "Importance", ascending=False
    )

    # Save data for LaTeX
    save_data_for_latex(
        {
            "parameter": df_imp["Parameter"].tolist(),
            "importance": df_imp["Importance"].tolist(),
        },
        "param_importances",
        dirs["data_importances"],
    )

    # Create bar chart visualization
    plt.figure(figsize=PLOT_CFG.importance_size)
    # Plot bars with parameter names on x-axis and importance values on y-axis
    plt.bar(df_imp["Parameter"], df_imp["Importance"], edgecolor="black")
    # Rotate parameter names for better readability
    plt.xticks(rotation=45, ha="right", fontsize=PLOT_CFG.x_tick_fs)
    plt.ylabel(PLOT_CFG.importance_ylabel)  # Importance score on y-axis
    plt.title(PLOT_CFG.importance_title, pad=PLOT_CFG.title_pad)  # Descriptive title
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    # Save with high resolution
    plt.savefig(os.path.join(dirs["figs"], "params_importances.pdf"))
    plt.close()  # Close figure to free memory


