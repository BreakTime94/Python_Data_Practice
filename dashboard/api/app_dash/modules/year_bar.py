import plotly.express as px


def year_bar_graph(df):
  year_group = (df.groupby("year", as_index=False)["salesAmount"]
                .sum()
                .sort_values(by="salesAmount", ascending=True)
                )
  fig_bar = px.bar(
    year_group,
    x="year",
    y="salesAmount",
  )
  return fig_bar