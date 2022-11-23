from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Replacer(BaseEstimator, TransformerMixin):
    """Replace values per column.

    Attributes:
        - mapper (Dict): a dict of dictionaries whose keys are the column
        names. The values are dictionaries whose keys are the original values
        to be replaced and the values are the new assigned ones.
    """

    def __init__(self, mapper: Dict) -> None:
        self.mapper = mapper

    def _replace_values(self, df: pd.DataFrame) -> pd.DataFrame:
        for column, mapping_dict in self.mapper.items():
            df.loc[:, column] = df.loc[:, column].replace(mapping_dict)
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
        df = self._replace_values(X)
        return pd.DataFrame(df, columns=self.column_names)
