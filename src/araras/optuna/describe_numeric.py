from typing import List
import pandas as pd

from .analyze import format_numeric_value


def describe_numeric(data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Generate descriptive statistics for numeric hyperparameters."""
    stats = []
    for col in cols:
        arr = data[col]
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
