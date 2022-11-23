"""Modules to scale data."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


class StandardDataFrameScaler(StandardScaler):
    """Scales and keeps column names from input DataFrame using StandardScaler.

    Check the scikit-learn official documentation for further information about
    the input parameters:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """  # noqa

    def __init__(self, copy=True, with_mean=True, with_std=True) -> None:
        """Initializes the standard scaler."""
        self.column_names = []
        super().__init__(
            copy=copy,
            with_mean=with_mean,
            with_std=with_std,
        )

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the scaler based on X.
        Args:
            X (pd.DataFrame): Input data.
            y (, optional): Ignored. Defaults to None.
        Returns:
            StandardDataFrameScaler: instance fitted.
        """
        super().fit(X)
        self.column_names = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scales X and adds column names.
        Args:
            X (pd.DataFrame): Input data.
        Returns:
            pd.DataFrame: scaled data.
        """
        assert str(X.columns) == str(self.column_names), (
            f"Columns don't have same order/elements. "
            f"Valid order: {self.column_names}"
        )

        X_scaled = super().transform(X)
        return pd.DataFrame(X_scaled, columns=self.column_names)

    def get_feature_names_out(self, input_features=None):
        return self.column_names
