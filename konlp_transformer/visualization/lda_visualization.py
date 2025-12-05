import os
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pca_visualization import font_prop

base_path = os.path.dirname(os.path.dirname(__file__))

def lda_visualization(df):
  X = np.vstack(df['synopsis_vectors'].values)
  Y = np.vstack(df['labels'].values)
  lda = LDA(n_components=2)
  lda_results = lda.fit_transform(X, Y)
  df['lda_x'] = list(lda_results[:, 0])
  df['lda_y'] = list(lda_results[:, 1])

  plt.figure(figsize=(10, 10))
  sns.scatterplot(x='lda_x', y='lda_y', data=df, hue='labels', palette="viridis", s=50, alpha=0.8)

  plt.title("LDA synopsis visualization cluster: 4개", fontproperties=font_prop, fontsize=20)

  plt.xlabel("LDA component_1", fontproperties=font_prop, fontsize=12)
  plt.ylabel("LDA component_2", fontproperties=font_prop, fontsize=12)
  plt.grid(True)
  save_path = os.path.join(base_path, "data/LDA_synopsis_visualization.png")
  plt.savefig(save_path)
  plt.show()
