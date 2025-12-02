import pandas as pd, numpy as np, pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def extract_data(df, minRating=4.0, custCount=200):

  user = df["user_id"].value_counts().reset_index().iloc[:custCount, :]
  movies = df["movie_id"].value_counts().reset_index()

  return df[(df["user_id"].isin(user["user_id"])) & (df["rating"] >= minRating)]

def svd_predict_model(users, degree=50):
  index = users["user_id"].unique()
  columns = users["movie_id"].unique()
  pivot_df = users.pivot_table(index = "user_id",columns="movie_id",values="rating", fill_value=None)

  means = pivot_df.mean(axis = 0)
  pivot_df.fillna(means, inplace = True)
  svd = TruncatedSVD(n_components = degree)
  user_svd = svd.fit_transform(pivot_df)
  matrix = svd.components_
  ratings_predict = user_svd@matrix
  df = pd.DataFrame(data = ratings_predict, index=index, columns = columns)

  #print(df)

  unpivot_df = df.stack().reset_index()
  unpivot_df.columns = ["user_id", "movie_id", "rating"]
  return unpivot_df

def performance_metrics(data):
  train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
  predict_df = svd_predict_model(train_data)

  comparison_df = pd.merge(predict_df, test_data, on=["user_id", "movie_id"], how="inner")
  actual_ratings = comparison_df["rating_y"]
  predicted_ratings = comparison_df["rating_x"]

  rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
  mae = mean_absolute_error(actual_ratings, predicted_ratings)
  return rmse, mae

if __name__ == "__main__":
  df = pd.read_pickle("../data/ratings.pkl")
  data = extract_data(df)
  rmse, mae = performance_metrics(data)
  print(rmse, mae)