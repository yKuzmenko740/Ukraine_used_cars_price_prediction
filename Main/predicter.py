from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib



class LGBMRegPredicter(ABC):
    """
    abstract class for predictors that uses lgb for prediction
    """

    @property
    @abstractmethod
    def _model_params(self):
        """
        dict with params for regression LGBM model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _train_params(self):
        """
        dict with training params
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, train_set: lgb.Dataset, test_set: lgb.Dataset = None):
        """
        Training the model and saving it in variable _model
        :param train_set:
        :param test_set:
        :return: none
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.array:
        """

        :param X: Dataframe with features
        :return: array with predicted values
        """

    @abstractmethod
    def set_train_params(self, params: dict):
        """
        :param params: dict with parameters for training the regression model
        :return: none
        """

    def set_model_params(self, params: dict):
        """

        :param params: dict with parameters for regression model
        :return: none
        """


class CarPricePredictor(LGBMRegPredicter):
    _train_params = None
    _model_params = None
    _model = lgb

    _NOT_FIT_ERROR = "Model is not fitted, invoke `fit` method"
    _is_fit = False

    def __init__(self, model_params: dict, training_params: dict):
        self._train_params = training_params
        self._model_params = model_params

    def fit(self, train_set: lgb.Dataset, test_set: lgb.Dataset = None):
        if 'valid_sets' in self._train_params.keys() and test_set is None:
            test_set = self._train_params['valid_sets']
            self._model = self._model.train(self._model_params, train_set,  **self._train_params)
        self._is_fit = True
        self._model = self._model.train(self._model_params, train_set, valid_sets=test_set, **self._train_params)

    def predict(self, X: pd.DataFrame) -> np.array:
        assert self._is_fit, self._NOT_FIT_ERROR
        return self._model.predict(X)


    def set_train_params(self, params: dict):
        self._train_params = params


    def save_model(self) :
        """
        Saving model into pickle file
        :return:
        """
        joblib.dump(self._model, 'lgb_car_price.pkl')