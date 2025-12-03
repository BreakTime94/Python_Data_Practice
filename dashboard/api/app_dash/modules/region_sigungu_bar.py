import plotly.express as px

def region_sigungu_bar_graph(df, selected_region):
  region_options = [{"label": region, "value": region} for region in df["region"].drop_duplicates().tolist()]

  sigungu = df[df["region"] == selected_region].groupby("sigungu", as_index=False)["salesAmount"].sum()

  fig_bar_sigungu = px.bar(
    sigungu,
    x="sigungu",
    y="salesAmount",
  )

  return region_options, fig_bar_sigungu