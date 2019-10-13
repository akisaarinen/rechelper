import numpy as np
import sklearn.metrics

def mae(y_true, y_pred):
  return sklearn.metrics.mean_absolute_error(y_true, y_pred)

def mse(y_true, y_pred):
  return sklearn.metrics.mean_squared_error(y_true, y_pred)

def rmse(y_true, y_pred):
  return np.sqrt(mse(y_true, y_pred))