import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

class DataSet:
  def __init__(self, ratings, user_idx_map, idx_user_map, item_idx_map, idx_item_map):
    self.ratings = ratings
    self.user_idx_map = user_idx_map
    self.idx_user_map = idx_user_map
    self.item_idx_map = item_idx_map
    self.idx_item_map = idx_item_map
    self.unique_users = np.unique(ratings["user"]).shape[0]
    self.unique_items = np.unique(ratings["item"]).shape[0]

  def print_stats(self):
    sparsity = self.ratings.shape[0] / (self.unique_users * self.unique_items)
    print("Ratings:        %d" % self.ratings.shape[0])
    print("Unique users:   %d" % self.unique_users)
    print("Unique items:   %d" % self.unique_items)
    print("Users/item:     %.2f" % (self.unique_users / self.unique_items))
    print("Sparsity:       %0.2f%%" % (100.0*sparsity))
    
def create(df,
    user_col="userId",
    item_col="movieId",
    rating_col="rating"
    ):
  user_idx, user_idx_map, idx_user_map = zero_based_array(df, user_col)
  item_idx, item_idx_map, idx_item_map = zero_based_array(df, item_col)
  rating_df = pd.DataFrame({
    "user": user_idx,
    "item": item_idx,
    "rating": df[rating_col]
  })
  return DataSet(rating_df, user_idx_map, idx_user_map, item_idx_map, idx_item_map)

def zero_based_array(df, col):
  values = df[col].values

  unique_values = np.unique(values)
  n_unique_values = unique_values.shape[0]
  max_value = np.max(unique_values)

  zero_based_map = np.zeros(max_value+1, dtype=int)
  zero_based_map[unique_values] = np.arange(n_unique_values)
  zero_based_values = zero_based_map[values]

  return zero_based_values, unique_values, zero_based_map
