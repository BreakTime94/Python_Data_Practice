import os
import pandas as pd, numpy as np
import umap.umap_ as UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE

# Windows 기본 한글 폰트 (맑은 고딕)
font_path = 'C:/Windows/Fonts/malgun.ttf'  # 또는 malgunbd.ttf(볼드)

font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

base_path = os.path.dirname(os.path.dirname(__file__))

def visualizations_tsne(
        df,
        pre_pca_components=50,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        hue=None,  # 예: 'label'
        auto_cluster=False,  # hue 컬럼이 없으면 KMeans로 자동 생성할지
        n_clusters=4,
        save_path="data/synopsis_tsne_scatter.png"
):
  embedding_matrix = X = np.vstack(df["synopsis_vectors"].values)

  if pre_pca_components is not None and pre_pca_components < embedding_matrix.shape[1]:
    pca = PCA(n_components=pre_pca_components, random_state=random_state)
    embedding_matrix = pca.fit_transform(embedding_matrix)

  # hue 컬럼이 필요한데 없으면 자동 생성
  if hue is not None and hue not in df.columns:
    if auto_cluster:
      km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
      df[hue] = km.fit_predict(embedding_matrix)
    else:
      print(f"[warn] '{hue}' 컬럼이 df에 없습니다. 단색으로 그립니다.")
      hue = None

  tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    max_iter= n_iter,
    learning_rate="auto",
    init="pca",
    random_state=random_state,
  )
  tsne_result = tsne.fit_transform(embedding_matrix)
  df["tsne_x"], df["tsne_y"] = tsne_result[:, 0], tsne_result[:, 1]

  X = df[["tsne_x","tsne_y"]]
  kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
  kmeans.fit(X)
  # 라벨링하기
  df['labels'] = kmeans.labels_
  in_path = os.path.join(base_path, "data/synopsis_embedding_label_df.pkl")
  df.to_pickle(in_path)

  plt.figure(figsize=(12, 10))
  if hue is None:
    plt.scatter(df["tsne_x"], df["tsne_y"], alpha=0.5, s=30)
  else:
    sns.scatterplot(data=df, x="tsne_x", y="tsne_y", hue=hue, s=30, alpha=0.7, edgecolor=None)
    plt.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")

  plt.title("synopsis 2차원 시각화 (t-SNE)", fontproperties=font_prop, fontsize=12)
  plt.xlabel("t-SNE Component 1");
  plt.ylabel("t-SNE Component 2")

  out_path = os.path.join(base_path, save_path)
  plt.grid(True);
  plt.tight_layout();
  plt.savefig(out_path, dpi=200);
  plt.show()


def visualizations_umap(
        df,
        pre_pca_components=50,  # 사전 PCA 축소 차원 (None이면 생략)
        n_neighbors=15,  # 이웃 수 (지역/전역 구조 균형)
        min_dist=0.1,  # 군집 뭉침 정도 (작을수록 조밀)
        metric="cosine",  # 문장 임베딩엔 보통 'cosine' 권장
        random_state=42,
        hue=None,  # 색상 기준 컬럼명 (예: 'label')
        auto_cluster=False,  # hue가 없으면 KMeans로 자동 생성할지
        n_clusters=4,  # auto_cluster 시 군집 개수
        save_path="data/synopsis_umap_scatter.png"
):
  # ----- 1. 임베딩 행렬 준비 -----
  X = np.vstack(df["synopsis_vectors"].values)

  # (선택) 사전 PCA로 축소 → 속도/안정성 향상
  if pre_pca_components is not None and pre_pca_components < X.shape[1]:
    pca = PCA(n_components=pre_pca_components, random_state=random_state)
    X = pca.fit_transform(X)

  # ----- 2. hue 컬럼 체크/자동 생성 -----
  if hue is not None and hue not in df.columns:
    if auto_cluster:
      km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
      df[hue] = km.fit_predict(X)
    else:
      print(f"[warn] '{hue}' 컬럼이 df에 없습니다. 단색으로 그립니다.")
      hue = None

  # ----- 3. UMAP 변환 -----
  umap = UMAP.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric=metric,
    random_state=random_state,
  )
  Z = umap.fit_transform(X)
  df["umap_x"], df["umap_y"] = Z[:, 0], Z[:, 1]

  # ----- 4. 한글 폰트 (macOS) -----

  # ----- 5. 시각화 -----
  plt.figure(figsize=(12, 10))
  if hue is None:
    plt.scatter(df["umap_x"], df["umap_y"], s=30, alpha=0.6)
  else:
    sns.scatterplot(
      data=df, x="umap_x", y="umap_y",
      hue=hue, s=30, alpha=0.7, edgecolor=None, palette="viridis"
    )
    plt.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")

  plt.title("synopsis 2차원 시각화 (UMAP)", fontproperties=font_prop, fontsize=12)
  plt.xlabel("UMAP 1");
  plt.ylabel("UMAP 2")
  base_path = os.path.dirname(os.path.dirname(__file__))
  out_path = os.path.join(base_path, save_path)
  plt.grid(True);
  plt.tight_layout();
  plt.savefig(out_path, dpi=200);
  plt.show()


def pca_kmeans():
  base_path = os.path.dirname(os.path.dirname(__file__))
  out_path = os.path.join(base_path, "data/synopsis_embedding_label_df.pkl")
  df = pd.read_pickle(out_path)
  vectors = np.vstack(df["synopsis_vectors"].values)
  # clustering
  pca = PCA(n_components=2)
  pca_results = pca.fit_transform(vectors)
  X = pd.DataFrame(pca_results, columns=["PCA_1", "PCA_2"])
  kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
  kmeans.fit(X)

  df['labels'] = kmeans.labels_
  in_path = os.path.join(base_path, "data/synopsis_embedding_label_df.pkl")
  df.to_pickle(in_path)

  # visualization
  plt.figure(figsize=(8, 8))
  plt.scatter(pca_results[:, 0], pca_results[:, 1], c='blue', s=1)
  plt.title("synopsis 2차원 시각화", fontproperties=font_prop, fontsize=16)
  plt.xlabel("PCA_component_1", fontproperties=font_prop, fontsize=12)
  plt.ylabel("PCA_component_2", fontproperties=font_prop, fontsize=12)
  plt.grid(True)
  save_path = os.path.join(base_path, "data/synopsis_scatter.png")
  plt.savefig(save_path)
  plt.show()

if __name__ == "__main__":
  pca_kmeans()