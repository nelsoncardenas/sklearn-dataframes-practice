from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HighCardinalityDroppper(BaseEstimator, TransformerMixin):
    """Drops high cardinality columns.

    Attributes:
        - threshold (float): numbers unique categories allowed per column
        expressed as the fraction respect to the number of rows.
        - exclude (list): list of columns which won't pass through this
        estimator.
    """

    def __init__(self, threshold: float = 0.9, exclude: List = []) -> None:
        self.threshold = threshold
        self.exclude = exclude

    def _columns_dropper(self, df: pd.DataFrame) -> None:
        nrows = df.shape[0]

        num_uniques = df.nunique().to_frame(name="num_uniques")

        missing_vals = num_uniques.assign(
            frac_uniques=num_uniques["num_uniques"] / nrows
        )

        missing_vals = missing_vals[
            ~missing_vals.index.isin(["incident_date", "policy_bind_date"])
        ]

        columns_to_drop = missing_vals[
            missing_vals["frac_uniques"] >= self.threshold
        ].index.values.tolist()

        self.selected_columns = df.columns.difference(columns_to_drop)

    def get_columns(self) -> List[str]:
        """Gets the list of remaining columns after the estimator is applied.

        Returns:
            List[str]: list of non-dropped columns.
        """
        return self.selected_columns.tolist()

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the values to replace by using 'transform' method.
        Args:
            X (pd.DataFrame): input data
        """
        self._columns_dropper(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Executes the methods to transform each type of column.
        Args:
            X (pd.DataFrame): input dataframe
        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        return X[self.selected_columns]
