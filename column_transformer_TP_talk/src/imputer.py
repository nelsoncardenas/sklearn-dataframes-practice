"""Module to impute null values in the input data."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class SimpleDataFrameImputer(SimpleImputer):
    """Imputes null values in the input data."""

    def __init__(
        self,
        missing_values=np.nan,
        strategy="median",
        fill_value=None,
        verbose="deprecated",
        copy=True,
        add_indicator=False,
    ) -> None:
        """Initializes the columns by category.
        Args:
            - categorical_columns (list): Categorical columns.
            - numerical_columns (list): Numerical columns.
            - text_columns (list): Text columns.
        """
        if strategy == "mode":
            strategy = "most_frequent"

        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
        )

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the values to replace by using 'transform' method.
        Args:
            X (pd.DataFrame): input data
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
