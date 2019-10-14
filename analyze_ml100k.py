import rechelper.movielens
import rechelper.dataset
import rechelper.baseline
import rechelper.metrics
import sys

from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Download
name="ml-latest-small"
rechelper.movielens.download_and_extract(name)
ratings_all,movies = rechelper.movielens.load_movielens(name)

# Create dataset
ds = rechelper.dataset.create(ratings_all)
print("Original dataframe")
ds.print_stats()

print("----")
print("Selected movies")
selected = ds
#selected = selected.min_counts(user_min_count=100, item_min_count=100)
selected.print_stats()

def item_by_title(title):
  movie_id = movies[movies['title'].str.find(title) >= 0]['movieId'].values[0]
  return selected.idx_item_map[movie_id]

df_train, df_test = train_test_split(selected.ratings, test_size=0.2)

y_true = df_test["rating"]

baselineMean = rechelper.baseline.BaselineMean(df_train, selected.unique_users, selected.unique_items)
y_all_mean = baselineMean.predict_all_mean(df_test)
y_item_mean = baselineMean.predict_item_mean(df_test)
y_user_mean = baselineMean.predict_user_mean(df_test)
y_both_mean = baselineMean.predict_item_user_mean(df_test)
print("=== Mean ===")
print("mean(all)         RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_all_mean)))
print("mean(item)        RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_item_mean)))
print("mean(user)        RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_user_mean)))
print("mean(item+user),  RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_both_mean)))

print("=====")
print("Calculating similarity matrix...")
baselineSim = rechelper.baseline.BaselineSim(
  df_train,
  n_users=selected.unique_users,
  n_items=selected.unique_items,
  min_sim=0.05,
  min_overlap=10,
)
print("=> Done!")

print("=========")
print("Showing a few example predictions")

def recs_for(find_title):
  item_idx = item_by_title(find_title)
  movie_id = selected.item_idx_map[item_idx]
  movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
  print("")
  print("Recs for: %s " % movie_title)
  print("")

  recs = baselineSim.predict_similar(item_idx)
  recs["movie_id"] = recs['item'].map(lambda x: selected.item_idx_map[x])
  recs["title"] = recs['movie_id'].map(lambda x: movies[movies["movieId"] == x]['title'].values[0])
  print(recs)

recs_for("Aladdin")
recs_for("Star Trek")
recs_for("Star Wars")

# Helpers
def save_sparse_csr(filename, array):
  np.savez(filename, data=array.data, indices=array.indices,
    indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
  loader = np.load(filename + '.npz')
  return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                    shape=loader['shape'])

# Enable when needed
if True:
  print("=========")
  print("Mapping imdb to movie ids for saving")

  item_idx = np.arange(selected.unique_items)
  movie_ids = map(lambda x: selected.item_idx_map[x], item_idx)
  movie_id_to_imdb_id = movies.set_index("movieId")["imdbId"].to_dict()
  imdb_ids = list(map(lambda x: movie_id_to_imdb_id[x], movie_ids))

  print("=========")
  print("Saving to file")
  save_sparse_csr("sim_%s_matrix"%name, baselineSim.cor)
  pd.DataFrame({
    'imdb_id': imdb_ids
  }).to_csv("sim_%s_mapping.csv" % name)
  print("Done.")