import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
  iris = load_iris()
  X = iris.data
  y = iris.target
  # X 행렬 정규화(통계적 거리, 마할라노비스)
  X_normalized = StandardScaler().fit_transform(X)
  # print(X_normalized)
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X_normalized)
  # print(X_pca)
  plt.figure(figsize=(5, 5))
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('Principal Component PCA')
  plt.show()