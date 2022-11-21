"""Module to drop nan columns in the input data."""
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HighCardinalityDroppper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.9, exclude: List = []) -> None:
        self.threshold = threshold
        self.exclude = exclude

    def _columns_dropper(self, df: pd.DataFrame) -> pd.DataFrame:
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

        return df[self.selected_columns]

    def get_columns(self) -> List:
        return self.selected_columns.tolist()

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the values to replace by using 'transform' method.
        Args:
            df (pd.DataFrame): input data
        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Executes the methods to transform each type of column.
        Args:
            df (pd.DataFrame): input dataframe
        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        df = self._columns_dropper(X)
        return df
