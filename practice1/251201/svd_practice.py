import numpy as np, pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from  scipy.sparse.linalg import svds
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv('../data/ratings.dat', sep="::", engine='python')
  # df.columns = ["user_id", "movie_id", "rating", "timestamp"]
  # df.drop('timestamp', axis=1, inplace=True)
  # df.to_pickle('../data/ratings.pkl')
  # df = pd.read_pickle("../data/ratings.pkl")
  # user = df["user_id"].value_counts().reset_index().iloc[:200, :]
  # movies = df["movie_id"].value_counts().reset_index()
  #
  # data = df[(df["user_id"].isin(user["user_id"])) & (df["rating"] >= 4)]
  #
  # pivot_df = pd.pivot_table(df, index="user_id", columns="movie_id", values="rating", aggfunc="mean")
  # pivot_df.to_pickle("../data/pivot_ratings.pkl")

  # df = pd.read_pickle("../data/pivot_ratings.pkl")
  # means = df.mean(axis=0) # 안 본 영화에 대해서는 평균 값을 계산해서 임의로 매긴다 -> 통계 낼 때 영향을 안 주므로.
  # # print(len(means))
  # df.fillna(means, inplace=True)
  #
  # df.to_pickle("../data/pivot_ratings_with_means.pkl")
if __name__ == "__main__":
  X = pd.read_pickle("../data/pivot_ratings_with_means.pkl").values
  
  U, S, VT = np.linalg.svd(X, full_matrices=False)
  svd = TruncatedSVD(n_components=2)
  A_reduced = svd.fit_transform(X)
  print(A_reduced.shape)

  U, S, VT = svds(X, k=5)
  print(S)
  D= np.diag(S)
  print(D)

  X_new_ratings = U@D@VT

  print(X_new_ratings)


