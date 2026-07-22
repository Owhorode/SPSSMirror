from typing import Tuple, List
import numpy as np
import pandas as pd
import patsy


class ParsedFormula:
    """
    Result of parsing an R-like formula string against a dataframe.
    Holds numpy arrays + readable term names — the patsy objects themselves
    are not exposed to library users.
    """

    def __init__(self, y: np.ndarray, X: np.ndarray, y_name: str,
                 x_names: List[str], n_obs: int):
        self.y = y
        self.X = X
        self.y_name = y_name
        self.x_names = x_names
        self.n_obs = n_obs

    def __repr__(self) -> str:
        return f"ParsedFormula(y='{self.y_name}', x={self.x_names}, n={self.n_obs})"


def parse_formula(formula: str, data: pd.DataFrame) -> ParsedFormula:
    """
    Parse an R-like formula, e.g. 'score ~ group + age', 'y ~ x1 * x2',
    'y ~ C(group) + x'. Returns clean numpy arrays with rows containing any
    NaNs in the involved columns automatically dropped (patsy default).
    """
    if "~" not in formula:
        raise ValueError(
            f"Invalid formula '{formula}'. Expected R-like syntax, e.g. 'y ~ x1 + x2'."
        )

    try:
        y_dm, x_dm = patsy.dmatrices(formula, data=data, return_type="dataframe")
    except patsy.PatsyError as e:
        raise ValueError(f"Could not parse formula '{formula}': {e}") from e

    y_name = y_dm.columns[0]
    x_names = list(x_dm.columns)

    y_arr = y_dm.to_numpy().ravel()
    x_arr = x_dm.to_numpy()

    if len(y_arr) < 2:
        raise ValueError(
            f"Formula '{formula}' produced fewer than 2 valid rows after dropping "
            f"missing values. Check that your columns have non-null overlapping data."
        )

    return ParsedFormula(y=y_arr, X=x_arr, y_name=y_name, x_names=x_names, n_obs=len(y_arr))


def parse_formula_no_intercept(formula: str, data: pd.DataFrame) -> ParsedFormula:
    """Same as parse_formula but strips the intercept column (for ANOVA-style designs)."""
    parsed = parse_formula(formula, data)
    if parsed.x_names and parsed.x_names[0] == "Intercept":
        return ParsedFormula(
            y=parsed.y, X=parsed.X[:, 1:], y_name=parsed.y_name,
            x_names=parsed.x_names[1:], n_obs=parsed.n_obs,
        )
    return parsed
