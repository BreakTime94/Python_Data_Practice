
import numpy as np
import pandas as pd


if __name__ == "__main__":
  df = pd.read_csv('data/amazon.csv')
  # df.info()  info에서 object는 String이다.
  df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
  # print(df["rating"])

  df["user_name"] = df["user_name"].str.split(",")
  df_explode = df.explode("user_name")
  df_explode = df_explode.dropna(subset=["rating"])
  # df_explode.info()

  # print(df_explode["user_name"])
  #
  #
  # # print(df.loc[0, "user_name"])
  # df.info()
  # print(df_explode["rating"])
  # exit()
  # pivot_table = pd.pivot_table(df_explode, values="rating", columns="product_name", index="user_name", aggfunc="sum") 얘도 같은 문법

  pivot_table = df_explode.pivot_table(values="rating", columns="product_name", index="user_name", aggfunc="mean", fill_value=np.nan)

  means = pivot_table.mean(axis=0)
  #print(pivot_table)
  print(means)
  pivot_table = pivot_table.fillna(means)

  print(pivot_table.iloc[:, 3].to_list())