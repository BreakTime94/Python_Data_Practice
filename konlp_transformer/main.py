import os
import ast
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import torch
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from torch.cuda import device
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from visualization.pca_visualization import pca_kmeans, font_prop, visualizations_tsne, visualizations_umap


def preprocessing_embeddings(df):
  device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
  df['genre'] = df['genre'].apply(lambda row: ast.literal_eval(row))
  embedder = SentenceTransformer("sentence-transformers/xlm-r-base-en-ko-nli-ststb", device=device)
  X_embeddings = embedder.encode(df['synopsis'].tolist(), convert_to_tensor=False, show_progress_bar=True)
  X_embeddings = normalize(X_embeddings)
  df['synopsis_vectors'] = list(X_embeddings)
  df.to_pickle("data/synopsis_embedding_df.pkl")

def put_label_max_count_genre(df):
  # 인덱스로 그룹바이 한 후에 각 인덱스 별로 가장 많이 나오는 장르를 counting 해줌
  genre_series = df.groupby("labels")["genre"].sum().apply(Counter)
  # 인덱스로 그룹바이 한 df에 대하여 각 장르의 랭킹을 부여해줌
  genre_rank_series = genre_series.apply(lambda row: dict(row.most_common()))
  df['genre_rank_series'] = df['labels'].map(genre_rank_series)

if __name__ == '__main__':
  df = pd.read_pickle("data/synopsis_rank_genre_embedding_label_df.pkl")
  X = np.vstack(df['synopsis_vectors'].values)
  Y = np.vstack(df['labels'].values)
  X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size= 0.2, random_state= 42, stratify=Y
  )
  svm_model = SVC(kernel="rbf", C=1.0, random_state= 42, probability=True)
  svm_model.fit(X_train, Y_train)

  Y_pred = svm_model.predict(X_test)
  accuracy = accuracy_score(Y_test, Y_pred)

  print("정확도", accuracy)











