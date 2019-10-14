import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

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


def normalize_per_user(x):
    x = x.astype(float)
    x_sum = x.sum()
    x_num = x.astype(bool).sum()
    x_mean = x_sum / x_num
    if x_num == 1 or x.std() == 0:
        return 0.0
    return (x - x_mean) / (x.max() - x.min())

class BaselineSim:
  def __init__(self, df_train, n_users, n_items, min_sim, min_overlap):
    print("[baseline-sim] Counting averages")
    self.item_averages = np.zeros(n_items, dtype=float)
    self.item_counts   = np.zeros(n_items, dtype=int)

    actual_avg = df_train.groupby('item')['rating'].mean()
    actual_cnt = df_train.groupby('item')['rating'].count()

    self.item_averages[actual_avg.index] = actual_avg
    self.item_counts[actual_cnt.index] = actual_cnt

    # Remove per-user mean from ratings before similarity
    print("[baseline-sim] Normalizing ratings")
    normalized_rating = df_train.groupby('user')['rating'].transform(normalize_per_user)

    print("[baseline-sim] Constructing COO matrix")
    coo = coo_matrix(
      (normalized_rating, (df_train['item'], df_train['user'])),
      shape=(n_items, n_users)
    )

    print("[baseline-sim] Calculating Cosine Similarity")
    cor = cosine_similarity(coo, dense_output=False)
    cor = cor.multiply(cor >= min_sim)
    coo_binary = coo.astype(bool).astype(int)

    print("[baseline-sim] Calculating Item Overlaps")
    item_overlaps = coo_binary.dot(coo_binary.transpose())
    cor = cor.multiply(item_overlaps >= min_overlap)

    self.cor = cor
    self.item_overlaps = item_overlaps
    print("[baseline-sim] Init complete")

  def predict_similar(self, item, top_k=10, not_in_list=[]):
    sim = pd.DataFrame({
      'item': np.arange(self.cor.shape[0]),
      'sim': self.cor[item].toarray().flat,
      'olap': self.item_overlaps[item].toarray().flat,
      'avg': self.item_averages,
      'cnt': self.item_counts,
    })
    # Drop item itself and all others requested
    to_drop = [item] + not_in_list
    sim = sim.drop(index=to_drop)
    sim = sim.sort_values(by=['sim', 'olap'], ascending=False)
    return sim[:top_k]
