import plotly.express as px

def category_product_treemap_graph(df, selected_category):
  treemap_options = None
  treemap_df = None

  if selected_category == "categoryName":
    treemap_options = "대분류"
    path = ["categoryName"]
    treemap_df = df.groupby(path, as_index=False)['salesAmount'].sum()
  elif selected_category == "productCategoryName":
    treemap_options = "제품분류"
    path = ["categoryName", "productCategoryName"]
    treemap_df = df.groupby(path, as_index=False)['salesAmount'].sum()
  else:
    treemap_options = "제품"
    path = ["categoryName", "productCategoryName", "productName"]
    treemap_df = df.groupby(path, as_index=False)['salesAmount'].sum()

  fig_treemmap = px.treemap(
    treemap_df,
    path=path,
    values="salesAmount",
    color="salesAmount",
  )

  return treemap_options, fig_treemmap