import urllib.request
import zipfile
import os.path
import pandas as pd
import numpy as np

DEFAULT_LOCAL_DATA_DIR="./data"

def load_ratings(dataset_name, dataset, local_data_dir):
  raw = pd.read_csv(
    "%s/%s/%s/%s" % (
      local_data_dir,
      dataset_name,
      dataset["extracted_path"],
      dataset["ratings"]["filename"]),
    sep=dataset["ratings"]["sep"],
    encoding=dataset["encoding"],
    names=dataset["ratings"]["cols"],
    engine=dataset["csv_engine"] if "csv_engine" in dataset else "c"
  )
  return raw

def postprocess_movies_ml100k(raw):
  raw.drop(["videoReleaseDate", "imdbUrl"], axis=1, inplace=True)
  raw["year"] = raw['releaseDate'].apply(lambda x: str(x).split('-')[-1])
  def mark_genres(movies, genres):
    def get_random_genre(gs):
      active = [genre for genre, g in zip(genres, gs) if g==1]
      if len(active) == 0:
        return 'Other'
      return np.random.choice(active)
    def get_all_genres(gs):
      active = [genre for genre, g in zip(genres, gs) if g==1]
      if len(active) == 0:
        return 'Other'
      return '-'.join(active)
    movies['genre'] = [
        get_random_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
    movies['allGenres'] = [
        get_all_genres(gs) for gs in zip(*[movies[genre] for genre in genres])]
  
  mark_genres(raw, ML100k_genre_cols)
  raw.drop(ML100k_genre_cols, axis=1, inplace=True)
  raw.drop("releaseDate", axis=1, inplace=True)
  return raw

def postprocess_movies_latest(raw):
  raw["year"] = raw['title'].apply(lambda x: 
    x.split("(")[-1].split(")")[0]
  )
  def get_random_genre(gs):
    if len(gs) == 0:
      return 'Other'
    return np.random.choice(gs)

  raw['genre'] = [
      get_random_genre(gs) for gs in raw['genres'].map(lambda x: x.split("|"))
  ]
  raw['allGenres'] = [
      "-".join(gs) for gs in raw['genres'].map(lambda x: x.split("|"))
  ]
  raw.drop("genres", axis=1, inplace=True)
  return raw

def load_movies(dataset_name, dataset, local_data_dir):
  raw = pd.read_csv(
    "%s/%s/%s/%s" % (
      local_data_dir,
      dataset_name,
      dataset["extracted_path"],
      dataset["movies"]["filename"]),
    sep=dataset["movies"]["sep"],
    encoding=dataset["encoding"],
    names=dataset["movies"]["cols"],
    engine=dataset["csv_engine"] if "csv_engine" in dataset else "c"
  )
  if "postprocess" in dataset["movies"]:
    raw = dataset["movies"]["postprocess"](raw)
  return raw

def load_movielens(dataset_name, local_data_dir=DEFAULT_LOCAL_DATA_DIR):
  dataset=datasets[dataset_name]
  movies=load_movies(dataset_name, dataset, local_data_dir)
  ratings=load_ratings(dataset_name, dataset, local_data_dir)
  return ratings, movies

def download_and_extract(dataset_name, local_data_dir=DEFAULT_LOCAL_DATA_DIR):
  dataset=datasets[dataset_name]
  url="%s/%s" % (dataset["url_path"], dataset["zip_filename"])
  zip_path="%s/%s" % (local_data_dir, dataset["zip_filename"])
  output_path="%s/%s" % (local_data_dir, dataset_name)
  if os.path.exists(zip_path):
    print("Already downloaded for %s at %s" % (dataset_name, zip_path))
  else:
    print("Downloading %s from '%s'..." % (dataset_name, url))
    urllib.request.urlretrieve(url, zip_path, reporthook=download_reporthook)

  if os.path.exists(output_path):
    print("Already extracted for %s at %s" % (dataset_name, output_path))
  else:
    print("* Extracting to '%s'..." % output_path)
    zip_ref = zipfile.ZipFile(zip_path, "r")
    zip_ref.extractall(path=output_path)
    print("* ZIP extract done")

def download_reporthook(chunk_num, chunk_size, total_size):
  MB=1024.0 * 1024.0
  downloaded_MB=(chunk_num*chunk_size)/MB
  total_MB=total_size/MB
  print("* Progress: %.2f/%.2f MB (%.1f%%)" %
    (downloaded_MB, total_MB, 100.0*downloaded_MB/total_MB),
    end="\r")

ML100k_genre_cols=[
      "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
      "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
      "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
  ]

datasets = {
  "ml-100k": {
    "url_path": "http://files.grouplens.org/datasets/movielens",
    "zip_filename": "ml-100k.zip",
    "extracted_path": "ml-100k",
    "encoding": "latin-1",
    "ratings": {
      "filename": "u.data",
      "sep": "\t",
      "cols": ["userId", "movieId", "rating", "timestamp"],
    },
    "movies": {
      "filename": "u.item",
      "sep": "|",
      "cols": ["movieId", "title", "releaseDate", "videoReleaseDate", "imdbUrl"] + ML100k_genre_cols,
      "postprocess": postprocess_movies_ml100k,
    }
  },
  "ml-latest-small": {
    "url_path": "http://files.grouplens.org/datasets/movielens",
    "zip_filename": "ml-latest-small.zip",
    "extracted_path": "ml-latest-small",
    "encoding": "latin-1",
    "ratings": {
      "filename": "ratings.csv",
      "sep": ",",
      "cols": None,
    },
    "movies": {
      "filename": "movies.csv",
      "sep": ",",
      "cols": None,
      "postprocess": postprocess_movies_latest,
    }    
  },
  "ml-latest": {
    "url_path": "http://files.grouplens.org/datasets/movielens",
    "zip_filename": "ml-latest.zip",
    "extracted_path": "ml-latest",
    "encoding": "latin-1",
    "ratings": {
      "filename": "ratings.csv",
      "sep": ",",
      "cols": None,
    },
    "movies": {
      "filename": "movies.csv",
      "sep": ",",
      "cols": None,
      "postprocess": postprocess_movies_latest,
    }    
  },  
  "ml-1m": {
    "url_path": "http://files.grouplens.org/datasets/movielens",
    "zip_filename": "ml-1m.zip",
    "extracted_path": "ml-1m",
    "encoding": "latin-1",
    "csv_engine": "python",
    "ratings": {
      "filename": "ratings.dat",
      "sep": "::",
      "cols": ["userId", "movieId", "rating", "timestamp"],
    },
    "movies": {
      "filename": "movies.dat",
      "sep": "::",
      "cols": ["movieId", "title", "genres"],
      "postprocess": postprocess_movies_latest,
    }    
  },  
}