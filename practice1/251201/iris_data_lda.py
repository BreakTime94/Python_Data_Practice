import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
  iris = load_iris()
  X = iris.data
  y = iris.target
  # X 행렬 정규화(통계적 거리, 마할라노비스)
  X_normalized = StandardScaler().fit_transform(X)
  # print(X_normalized)
  lda = LDA(n_components=2)
  X_lda = lda.fit_transform(X_normalized, y)
  # print(X_pca)
  plt.figure(figsize=(5, 5))
  plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
  plt.xlabel('Linear Component 1')
  plt.ylabel('Linear Component 2')
  plt.title('Linear Discriminant Analysis LDA')
  plt.show()