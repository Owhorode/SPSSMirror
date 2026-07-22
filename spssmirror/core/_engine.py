from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


class DataEngine:
    """
    Single pandas-backed data engine for SPSSMirror. No backend switching —
    Polars support is intentionally excluded from this build.
    """

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}

    @staticmethod
    def _check_no_duplicate_columns(df: pd.DataFrame) -> None:
        """Duplicate column names cause `df[name]` to return a DataFrame
        instead of a Series, which breaks every engine's validation logic
        with a confusing AttributeError deep inside unrelated code. Catching
        it once here, at load time, gives a clear message instead."""
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        if dupes:
            raise ValueError(
                f"Duplicate column names are not supported: {dupes}. "
                f"Rename or drop the duplicates before loading."
            )

    def load_dict(self, data: Dict[str, List[Any]]) -> "DataEngine":
        if not data:
            raise ValueError("Input data cannot be empty.")
        lengths = {k: len(v) for k, v in data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"All columns must have equal length. Got {lengths}")
        df = pd.DataFrame(data)
        self._check_no_duplicate_columns(df)
        self._df = df
        self._metadata["n_rows"] = len(self._df)
        self._metadata["n_cols"] = len(self._df.columns)
        return self

    def load_dataframe(self, df: pd.DataFrame) -> "DataEngine":
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(df)}")
        self._check_no_duplicate_columns(df)
        self._df = df.copy()
        self._metadata["n_rows"] = len(self._df)
        self._metadata["n_cols"] = len(self._df.columns)
        return self

    def load_csv(self, filepath: str, **kwargs) -> "DataEngine":
        try:
            df = pd.read_csv(filepath, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: '{filepath}'")
        except Exception as e:
            raise ValueError(f"Could not read CSV file '{filepath}': {e}") from e
        self._check_no_duplicate_columns(df)
        self._df = df
        self._metadata["source"] = filepath
        self._metadata["n_rows"] = len(self._df)
        self._metadata["n_cols"] = len(self._df.columns)
        return self

    def load_excel(self, filepath: str, **kwargs) -> "DataEngine":
        try:
            df = pd.read_excel(filepath, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Excel file not found: '{filepath}'")
        except Exception as e:
            raise ValueError(f"Could not read Excel file '{filepath}': {e}") from e
        self._check_no_duplicate_columns(df)
        self._df = df
        self._metadata["source"] = filepath
        self._metadata["n_rows"] = len(self._df)
        self._metadata["n_cols"] = len(self._df.columns)
        return self

    def _require_loaded(self) -> pd.DataFrame:
        if self._df is None:
            raise ValueError("No data loaded. Call load_dict/load_dataframe/load_csv first.")
        return self._df

    def to_dataframe(self) -> pd.DataFrame:
        return self._require_loaded()

    def to_numpy(self, column: str) -> np.ndarray:
        df = self._require_loaded()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
        return df[column].to_numpy()

    def columns(self) -> Tuple[str, ...]:
        return tuple(self._require_loaded().columns)

    def dtypes(self) -> Dict[str, str]:
        df = self._require_loaded()
        return {c: str(t) for c, t in df.dtypes.items()}

    def preview(self, n_rows: int = 5) -> str:
        return self._require_loaded().head(n_rows).to_string()

    def shape(self) -> Tuple[int, int]:
        return self._require_loaded().shape

    def select_columns(self, columns: List[str]) -> "DataEngine":
        df = self._require_loaded()
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        self._df = df[columns]
        return self

    def get_column(self, name: str) -> pd.Series:
        df = self._require_loaded()
        if name not in df.columns:
            raise ValueError(f"Column '{name}' not found. Available: {list(df.columns)}")
        return df[name]

    def add_column(self, name: str, values) -> "DataEngine":
        df = self._require_loaded()
        df[name] = values
        return self

    def drop_nulls(self, subset: Optional[List[str]] = None) -> "DataEngine":
        df = self._require_loaded()
        self._df = df.dropna(subset=subset)
        return self

    def unique_values(self, column: str) -> List[Any]:
        return self.get_column(column).dropna().unique().tolist()

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
