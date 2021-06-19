import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def evaluateRegressor(true,predicted) -> np.array:
    """"
        get main metrics
        
        return: np.array with main metrics (mse, mae, rmse, logrmse)
    """
    MSE = mean_squared_error(true,predicted,squared = True)
    MAE = mean_absolute_error(true,predicted)
    RMSE = mean_squared_error(true,predicted,squared = False)
    LogRMSE = mean_squared_error(np.log(true),np.log(predicted),squared = False)
    return  np.array(MSE, MAE, RMSE, LogRMSE)
