"""Modules to encode data."""
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class OneHotDataFrameEncoder(OneHotEncoder):
    def __init__(
        self,
        categories: str = "auto",
        drop=None,
        sparse: bool = True,
        dtype: float = np.float,
        handle_unknown: str = "error",
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[Union[int, float]] = None,
    ) -> None:
        """Initializes the one hot encoder."""
        self.column_names = []
        super().__init__(
            categories=categories,
            drop=drop,
            sparse=sparse,
            dtype=dtype,
            handle_unknown=handle_unknown,
            min_frequency=min_frequency,
            max_categories=max_categories,
        )

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the scaler based on X.
        Args:
            X (pd.DataFrame): Input data.
            y (, optional): Ignored. Defaults to None.
        Returns:
            StandardDataFrameScaler: instance fitted.
        """
        self.ohe_scaler = super().fit(X)
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

        X_scaled = self.ohe_scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.column_names)

    def get_feature_names_out(self, input_features=None):
        return self.column_names