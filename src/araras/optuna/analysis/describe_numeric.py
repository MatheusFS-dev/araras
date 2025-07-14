"""
Module describe_numeric of analysis

Functions:
    - describe_numeric: Generate descriptive statistics for numeric hyperparameters.

Example:
    >>> from araras.optuna.analysis.describe_numeric import describe_numeric
    >>> describe_numeric(...)
"""
from araras.core import *
import pandas as pd

from .analyzer import format_numeric_value


def describe_numeric(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Generate descriptive statistics for numeric hyperparameters."""
    stats = []
    for col in cols:
        arr = data[col].dropna()  # Remove NaN values upfront

        if arr.empty:
            # Handle columns with no valid data
            formatted = {
                "Parameter": col,
                "Mean": "No data",
                "Std": "No data",
                "Median": "No data",
                "Min (25% quantile)": "No data",
                "Max (75% quantile)": "No data",
                "Min (5% quantile)": "No data",
                "Max (95% quantile)": "No data",
            }
        else:
            # Compute statistics on valid data only
            raw = {
                "Parameter": col,
                "Mean": arr.mean(),
                "Std": arr.std(),
                "Median": arr.median(),
                "Min (25% quantile)": arr.quantile(0.25),
                "Max (75% quantile)": arr.quantile(0.75),
                "Min (5% quantile)": arr.quantile(0.05),
                "Max (95% quantile)": arr.quantile(0.95),
            }
            formatted = {"Parameter": col}
            for k, v in raw.items():
                if k != "Parameter":
                    formatted[k] = format_numeric_value(v)

        stats.append(formatted)
    return pd.DataFrame(stats)
