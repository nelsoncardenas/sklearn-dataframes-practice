from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NaNColumnsDropper(BaseEstimator, TransformerMixin):
    """Drops columns with amount of NaN values greater than a given threshold.

    Attributes:
        - threshold (float): numbers NaN values allowed per column
        expressed as the fraction respect to the number of rows.
    """

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold

    def _columns_dropper(self, df: pd.DataFrame) -> None:
        nrows = df.shape[0]
        missing_vals = df.isna().sum().to_frame(name="num_nans")
        missing_vals = missing_vals.assign(
            frac_nans=missing_vals["num_nans"] / nrows
        )
        columns_to_drop = missing_vals[
            missing_vals["frac_nans"] >= self.threshold
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
        return pd.DataFrame(
            X[self.selected_columns], columns=self.selected_columns
        )
