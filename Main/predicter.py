from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import lightgbm as lgb

from lightgbm import LGBMRegressor

class LGBPredicter(ABC):
    """
    abstract class for predictors that uses lgb for prediction
    """
    pass