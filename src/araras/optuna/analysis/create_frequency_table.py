"""
Module create_frequency_table of analysis

Functions:
    - create_frequency_table: Generate frequency tables for categorical hyperparameters.

Example:
    >>> from araras.optuna.analysis.create_frequency_table import create_frequency_table
    >>> create_frequency_table(...)
"""
from araras.commons import *
import pandas as pd


def create_frequency_table(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Generate frequency tables for categorical hyperparameters."""
    rows = []
    for col in cols:
        counts = data[col].value_counts(normalize=True)
        for cat, frac in counts.items():
            rows.append({
                "Parameter": col,
                "Category": cat,
                "Fraction": round(frac, 4),
                "Count": int(data[col].value_counts()[cat]),
            })
    return pd.DataFrame(rows)
