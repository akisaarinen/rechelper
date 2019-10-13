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
#selected = selected.min_counts(10, 10)
selected.print_stats()

df_train, df_test = train_test_split(selected.ratings, test_size=0.2)
baseline = rechelper.baseline.BaselineMean(df_train, selected.unique_users, selected.unique_items)

y_true = df_test["rating"]
y_all_mean = baseline.predict_all_mean(df_test)
y_item_mean = baseline.predict_item_mean(df_test)
y_user_mean = baseline.predict_user_mean(df_test)
y_both_mean = baseline.predict_item_user_mean(df_test)

print("All mean,      RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_all_mean)))
print("Item mean,     RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_item_mean)))
print("User mean,      RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_user_mean)))
print("Item-User mean, RMSE: %.3f" % (rechelper.metrics.rmse(y_true, y_both_mean)))