import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import pairwise_distances

class BaselineMean:
  def __init__(self, df_train, n_users, n_items):
    coo = coo_matrix(
      (df_train['rating'], (df_train['user'], df_train['item'])),
      shape=(n_users, n_items)
    )
    # Calculate means
    with np.errstate(invalid='ignore'):
      self.mean_item_ratings = coo.sum(axis=0).getA1() / np.bincount(coo.col, minlength=n_items)
      self.mean_user_ratings = coo.sum(axis=1).getA1() / np.bincount(coo.row, minlength=n_users)
      self.mean_all = np.nanmean(coo.data)

    # Replace NaN with overall mean
    self.mean_item_ratings[np.isnan(self.mean_item_ratings)] = self.mean_all
    self.mean_user_ratings[np.isnan(self.mean_user_ratings)] = self.mean_all

  def predict_all_mean(self, df_test):
    return np.repeat(self.mean_all, df_test.shape[0])

  def predict_item_mean(self, df_test):
    return self.mean_item_ratings[df_test["item"]]

  def predict_user_mean(self, df_test):
    return self.mean_user_ratings[df_test["user"]]

  def predict_item_user_mean(self, df_test):
    return (self.predict_item_mean(df_test) + self.predict_user_mean(df_test))/2.0