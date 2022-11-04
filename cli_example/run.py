import logging

import mlflow.lightgbm
import pandas as pd
import typer
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


from modules.imputer import Imputer
from modules.scaler import StandardDataFrameScaler
from modules.enconder import OneHotDataFrameEncoder
from modules.column_transformer import ColumnDataFrameTransformer


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train(
    path_train_test: str = None,
    path_model: str = None,
    random_state: int = None,
) -> None:
    """Trains a model.

    Args:
        path_train_test (str): Path for train/test data.
        path_model (str): Path to save the trained model.
        random_state (int):  Seed used by the random number generator.
    """
    logger.debug("Input paths")
    with open("config.yml", "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    logger.info("Reading input data")
    input_columns = (
        config["train_columns_by_type"]["categorical_columns"]
        + config["train_columns_by_type"]["numerical_columns"]
        + config["train_columns_by_type"]["text_columns"]
        + [config["target_column"]]
    )
    path_train = path_train_test.replace("{split}", "train")
    path_val = path_train_test.replace("{split}", "val")

    df_train = pd.read_csv(path_train, usecols=input_columns)
    df_val = pd.read_csv(path_val, usecols=input_columns)

    logger.info("Creating column imputation step")
    imputer = Imputer(
        categorical_columns=config["train_columns_by_type"]["categorical_columns"],
        numerical_columns=config["train_columns_by_type"]["numerical_columns"],
        text_columns=config["train_columns_by_type"]["text_columns"],
        **config["imputer"],
    )

    logger.info("Creating column transformation step")
    numeric_transformer = StandardDataFrameScaler()
    one_hot_transformer = OneHotDataFrameEncoder(handle_unknown="ignore")

    numeric_columns = config["train_columns_by_type"]["numerical_columns"]
    one_hot_columns = (
        config["train_columns_by_type"]["categorical_columns"]
        + config["train_columns_by_type"]["text_columns"]
    )

    column_transformer = ColumnDataFrameTransformer(
        transformers=[
            ("numeric scaler", numeric_transformer, numeric_columns),
            ("cat_and_txt encoder", one_hot_transformer, one_hot_columns),
        ]
    )

    logger.info("Creating classifier step")
    classifier = LGBMClassifier(
        class_weight="balanced",
        random_state=0,
        learning_rate=0.06,
        n_estimators=2000,
        reg_lambda=0.19,
        reg_alpha=0.19,
    )

    logger.info("Creating model pipeline")
    preprocessor = Pipeline(
        [
            ("column_imputer", imputer),
            ("column_transformer", column_transformer),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    logger.info("Creating X and y samples")
    X_train = df_train.drop(columns=[config["target_column"]])
    y_train = df_train[config["target_column"]]

    X_val = df_val.drop(columns=[config["target_column"]])
    y_val = df_val[config["target_column"]]

    logger.info("Training the model")
    mlflow.lightgbm.autolog()
    with mlflow.start_run() as _:
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        logger.info("Evaluating the model using training data")
        f1_train = f1_score(
            y_true=y_train,
            y_pred=y_pred_train,
            pos_label=config["positive_label_value"],
        )
        logger.info("Evaluating the model using validation data")
        y_pred_val = model.predict(X_val)
        f1_val = f1_score(
            y_true=y_val, y_pred=y_pred_val, pos_label=config["positive_label_value"]
        )
        logger.info("...Logging in mlflow")
        mlflow.log_metric(key="f1_train", value=f1_train)
        mlflow.log_metric(key="f1_validation", value=f1_val)

    logger.info(f"...Training f1 score: {f1_train:.5f}")
    logger.info(f"...Validation f1 score: {f1_val:.5f}")

    logger.info("4_train finished")


if __name__ == "__main__":
    typer.run(train)
