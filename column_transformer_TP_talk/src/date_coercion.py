"""Module to replace some values in the input data."""
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateCoercion(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns: List) -> None:
        self.date_columns = date_columns

    def _caster(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self.date_columns:
            df[column] = pd.to_datetime(df.loc[:, column])
        return df

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the values to replace by using 'transform' method.
        Args:
            df (pd.DataFrame): input data
        """
        self.column_names = X.columns
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Executes the methods to transform each type of column.
        Args:
            df (pd.DataFrame): input dataframe
        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        df = self._caster(X)
        return pd.DataFrame(df, columns=self.column_names)
