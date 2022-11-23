"""Module to replace some values in the input data."""
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Filters the specified columns.

    Attributes:
        - selected_columns (list): list of columns which will be used for the
        Machine Learning model.
    """

    def __init__(self, selected_columns: List) -> None:
        self.selected_columns = selected_columns

    def _filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.selected_columns]

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
        df = self._filter_columns(X)
        return pd.DataFrame(df, columns=self.selected_columns)
