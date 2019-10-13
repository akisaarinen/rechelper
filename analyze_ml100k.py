import rechelper.movielens
import rechelper.dataset

# Download
name="ml-100k"
rechelper.movielens.download_and_extract(name)
ratings_all,movies = rechelper.movielens.load_movielens(name)
ds = rechelper.dataset.create(ratings_all)

ds.print_stats()