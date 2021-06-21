import pandas as pd
import logging
from config import Config
from data_preprocessing import DataReader
from metrics import evaluateRegressor
from predicter import CarPricePredictor
from sklearn.model_selection import train_test_split
from typing import Tuple
import json
import warnings

RANDOM_STATE = 17


# write user training and predicting pipelines

def prepare_test_train_df(conf: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reader = DataReader()
    full_df = reader.read_cars_data(conf)
    logging.info("Cars data is loaded")
    processed_df = reader.data_processing(full_df)
    train_df, test_df = train_test_split(processed_df, test_size=.2, random_state=RANDOM_STATE, shuffle=True)
    logging.info(
        f"Train and test datasets are created.\nTrain shape is {train_df.shape}\nTest shape is {test_df.shape}")
    return train_df, test_df


# def prepare_for_predict(conf: Config) -> pd.DataFrame:
#     reader = DataReader()
#     full_df = reader.read_cars_data(conf)
#     logging.info("Cars data is loaded")
#     return reader.data_processing(full_df)


def train_regressor(config: Config, train_df: pd.DataFrame, test_df: pd.DataFrame) -> CarPricePredictor:
    train_params = {
        "num_boost_round": 5000,
        'early_stopping_rounds': 500,
        'verbose_eval': 100,
        'valid_sets': DataReader.create_lgb_dataset(test_df)
    }
    model_params = load_json(config)
    train_set = DataReader.create_lgb_dataset(train_df)
    reg = CarPricePredictor(model_params, train_params)
    reg.fit(train_set)
    return reg


def load_json(config: Config) -> dict:
    with open(config.model_coefs, 'r') as f:
        return json.load(f)


def train_pipeline(conf: Config):
    """
        Performs that whole training pipeline:
        * reads and preprocesses training data
        * fits CarPricePredictor model
        * computes test metrics using metrics.py
        """
    # read data
    train_df, test_df = prepare_test_train_df(conf)
    clf = train_regressor(conf, train_df, test_df)
    metrics = evaluateRegressor(test_df.Price_USD, clf.predict(train_df.drop('Price_USD', axis=1)))
    logging.info(f"Clf metrics on test dataset: {metrics}")
    return clf, metrics


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    conf = Config.load_from_file()
    train_df, test_df = prepare_test_train_df(conf)
    train_df.to_csv('../Data/train_df.csv', index=False)
    test_df.to_csv('../Data/test_df.csv', index=False)
    # clf, metrics = train_pipeline(conf)
