import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def visualize_height_weight(df):
  plt.figure(figsize=(5,5))
  plt.scatter(df['Height(Inches)'], df['Weight(Pounds)']) # scatter는 점을 찍는 행위
  plt.xlabel('Height')
  plt.ylabel('Weight')
  plt.show()

def eigen_values_vectors(df):
  X = df[['Height(Inches)', 'Weight(Pounds)']].to_numpy() # numpy로 바꾸는건 필드를 없애고 행렬을 만드는 행위다. 행렬은 변수 선언을 대문자로 한다.
  cov_pivot = np.cov(X.T)
  eigen_values, eigenvectors = np.linalg.eig(cov_pivot)
  return eigen_values, eigenvectors

if __name__ == "__main__":
  # filename = "data/SOCR-HeightWeight.csv"
  # df = pd.read_csv(filename) # DataFrame
  # table = df[df["Weight(Pounds)"] >= 150]
  # print(table)

  lists = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Tf = 10 in lists
  print(Tf)

  exit()

  print(df.shape)
  visualize_height_weight(df)
  eigen_values, eigenvectors = eigen_values_vectors(df)

  print(eigen_values)
  print(eigenvectors)