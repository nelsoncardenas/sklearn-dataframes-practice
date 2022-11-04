"""Module to impute null values in the input data."""
import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Imputer(BaseEstimator, TransformerMixin):
    """Imputes null values in the input data."""

    def __init__(
        self,
        categorical_columns: list,
        numerical_columns: list,
        text_columns: list,
        categorical_mode: str,
        numerical_mode: str,
    ) -> None:
        """Initializes the columns by category.

        Args:
            categorical_columns (list): Categorical columns.
            numerical_columns (list): Numerical columns.
            text_columns (list): Text columns.
        """
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.categorical_mode = categorical_mode
        self.numerical_mode = numerical_mode

        self.filler_categorical = [None]
        self.filler_numerical = [None]
        self.filler_text = [None]

    def fit(self, df: pd.DataFrame, y=None):
        """Fits the values to replace by using 'transform' method.

        Args:
            df (pd.DataFrame): input data
        """
        self._fit_categorical(df)
        self._fit_numerical(df)
        return self

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Executes the methods to transform each type of column.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: Dataframe with imputed values.
        """
        df = self._impute_categorical(df)
        df = self._impute_numerical(df)
        df = self._impute_text(df)
        return df

    def _fit_categorical(self, df: pd.DataFrame) -> pd.Series:
        self._filler_categorical = self._general_fitter(
            df, self.categorical_columns, self.categorical_mode
        )

    def _fit_numerical(self, df) -> pd.Series:
        self._filler_numerical = self._general_fitter(
            df, self.numerical_columns, self.numerical_mode
        )

    def _general_fitter(
        self, df: pd.DataFrame, cols: list, strategy_key: str
    ) -> pd.Series:
        """Creates a filler parameter for each column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            cols (list): Columns to be imputed.
            strategy_key (str): strategy to generate the filler. Options: 'mode', 'median'.

        Returns:
            pd.Series: values to be replaced in each column.
        """
        strategies = {"mode": df[cols].mode, "median": df[cols].median}
        filler = strategies[strategy_key]()
        if strategy_key == "mode":
            filler = filler.iloc[0]
        return filler

    def _impute_categorical(self, df: pd.DataFrame):
        return self._general_impute(
            df, self.categorical_columns, self._filler_categorical
        )

    def _impute_numerical(self, df: pd.DataFrame):
        return self._general_impute(df, self.numerical_columns, self._filler_numerical)

    def _impute_text(self, df: pd.DataFrame):
        df[self.text_columns] = (
            df[self.text_columns]
            .isnull()
            .replace({True: "undefined", False: "defined"})
        )
        return df

    def _general_impute(
        self, df: pd.DataFrame, cols: list, filler: pd.Series
    ) -> pd.DataFrame:
        """Imputes null values based on cols and filler.

        Args:
            df (pd.DataFrame): Input data.
            cols (list): List of columns to fill.
            filler (pd.Series): Series of values to impute.

        Returns:
            pd.DataFrame: DataFrame with nulls imputed.
        """
        df[cols] = df[cols].fillna(filler)
        return df
