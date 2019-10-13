import rechelper.movielens
import rechelper.dataset
import rechelper.baseline
import rechelper.metrics

from sklearn.model_selection import train_test_split

# Download
name="ml-latest-small"
rechelper.movielens.download_and_extract(name)
ratings_all,movies = rechelper.movielens.load_movielens(name)
ds = rechelper.dataset.create(ratings_all)

print("Original dataframe")
ds.print_stats()

print("----")
print("Selected movies")
selected = ds
#selected = selected.min_counts(user_min_count=400, item_min_count=490)
selected.print_stats()

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