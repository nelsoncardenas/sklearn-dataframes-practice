"""Modules to encode data."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class OneHotDataFrameEncoder(BaseEstimator, TransformerMixin):
    """Encodes and keeps names of categorical features as a one-hot code structure."""

    def __init__(self, handle_unknown="ignore") -> None:
        """Initializes the encoder."""
        self.handle_unknown = handle_unknown
        self.one_hot_encoder = OneHotEncoder(handle_unknown=handle_unknown)
        self.column_names = []
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the one hot encoder based on X.

        Args:
            X (pd.DataFrame): Input data.
            y (, optional): Ignored. Defaults to None.

        Returns:
            OneHotDataFrameEncoder: instance fitted.
        """
        self.one_hot_encoder.fit(X)
        self.column_names = X.columns
        self.feature_names = self.one_hot_encoder.get_feature_names_out()
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

        X_encoded = self.one_hot_encoder.transform(X)

        return pd.DataFrame(
            X_encoded.toarray().astype("int8"), columns=self.feature_names
        )

    def get_feature_names_out(self, input_features=None):
        return self.feature_names
