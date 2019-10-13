import rechelper.movielens
import rechelper.dataset

# Download
name="ml-100k"
rechelper.movielens.download_and_extract(name)
ratings_all,movies = rechelper.movielens.load_movielens(name)
ds = rechelper.dataset.create(ratings_all)

print("Original dataframe")
ds.print_stats()

print("----")
print("Selected movies")
selected = ds.min_counts(250, 250)
selected.print_stats()

selected.ratings['itemId'] = selected.item_idx_map[selected.ratings['item']]
selected.ratings['userId'] = selected.user_idx_map[selected.ratings['user']]
print(selected.ratings)