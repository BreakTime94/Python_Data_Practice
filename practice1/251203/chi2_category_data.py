import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

critical_value = np.array([
    [0.05, 3.841],
    [0.01, 6.635],
    [0.001, 10.828]
])

critical_df = pd.DataFrame(critical_value,columns=["유의수준", "임계값"])


def diaper_beer_chi2_pract():
    observed = np.array([
        [30, 10],
        [5, 55]
    ])


    index = ["기저귀(O)", "기저귀(X)"]
    column = ["맥주(O)", "맥주(X)"]
    pivot_df = pd.DataFrame(observed, index=index, columns=column)

    chi2, p_value, _, _ = chi2_contingency(pivot_df)
    print(critical_df)
    print("--------------------------------")
    print("카이제곱 통계량: ", chi2)
    print("p_value: ", p_value)

def discount_rating_chi2_pract():
    df = pd.read_csv("../data/amazon_data.csv")
    df = df[["평점", "할인율"]]
    df['할인율'] = df["할인율"].str.replace("%", "", regex=False).astype(float)
    df['평점'] = pd.to_numeric(df['평점'], errors='coerce')
    df['할인율그룹'] = pd.cut(
        df['할인율'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0~20", "20~40", "40~60", "60~80", "80~100"],
        include_lowest=True,
    )

    df['평점그룹'] = pd.cut(
        df['평점'],
        bins=[0, 3.0, 4.0, 5.0],
        labels=["0~3.0", "3.0~4.0", "4.0~5.0"],
        include_lowest=True,
    )

    pivot_df = pd.crosstab(df["할인율그룹"], df["평점그룹"])
    print(pivot_df)
    chi2, p_value, _, _ = chi2_contingency(pivot_df)

    print(critical_df)
    print("-------------------------------------")
    print("카이제곱 통계량", chi2)
    print("p_value: ", p_value)


if __name__ == "__main__":
    discount_rating_chi2_pract()