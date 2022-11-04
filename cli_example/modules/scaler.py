"""Modules to scale data."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardDataFrameScaler(BaseEstimator, TransformerMixin):
    """Scales and keeps column names from input DataFrame using StandardScaler."""

    def __init__(self) -> None:
        """Initializes the scaler."""
        self.std_scaler = StandardScaler()
        self.column_names = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the scaler based on X.

        Args:
            X (pd.DataFrame): Input data.
            y (, optional): Ignored. Defaults to None.

        Returns:
            StandardDataFrameScaler: instance fitted.
        """
        self.std_scaler.fit(X)
        self.column_names = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scales X and adds column names.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: scaled data.
        """
        assert str(X.columns) == str(
            self.column_names
        ), f"Columns don't have same order/elements. Valid order: {self.column_names}"

        X_scaled = self.std_scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.column_names)

    def get_feature_names_out(self, input_features=None):
        return self.column_names
