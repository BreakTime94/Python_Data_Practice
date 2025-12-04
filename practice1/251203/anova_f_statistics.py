import pandas as pd
import numpy as np
import scipy.stats as stats

critical_value = np.array([
        [0.05, 3.841],
        [0.01, 6.635],
        [0.001, 10.828]
    ])

critical_df = pd.DataFrame(critical_value,columns=["유의수준", "임계값"])

if __name__ == "main":
    df = pd.read_csv("data/amazon_data.csv")
    df = df[["평점", "할인율"]]
    df['할인율'] = df["할인율"].str.replace("%", "", regex=False).astype(float)
    df['평점'] = pd.to_numeric(df['평점'], errors='coerce')
    df['할인율그룹'] = pd.cut(
        df['할인율'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0~20", "20~40", "40~60", "60~80", "80~100"],
        include_lowest=True,
    )
    df.dropna(subset=["평점", "할인율그룹"], inplace=True)

    # groups = df.groupby("할인율그룹")
    # for group_name, group_table in groups:
    #     print(group_name)
    #     print(group_table)
    # exit()
    dsct_groups = [group["평점"].values
                   for name, group in df.groupby("할인율그룹", observed=False)]
    print(len(dsct_groups))

    f_stat, p_val = stats.f_oneway(*dsct_groups)

    print(critical_df)
    print("---------------------------------")
    print("f-statistic:", f_stat)
    print("p-value:", p_val)
    print("f 통계량을 높을수록 서로 관련성이 없습니다.")
    print("서로 멀리 떨어져 있기 때문입나다.")
    print("공정에서는 f통계량이 작을수록 같은 제품을 만들고 있다고 생각할 수 있습니다.")