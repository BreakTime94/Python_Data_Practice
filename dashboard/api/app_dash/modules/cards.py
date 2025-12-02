

def bring_card_data(df):
  total_sales = round(df["salesAmount"].sum())
  total_profit = round(df["netProfit"].sum())
  total_customers = round(df["customerName"].nunique())
  total_qnty = round(df["quantity"].sum())
  return {"total_sales": total_sales, "total_profit": total_profit, "total_customers": total_customers, "total_qnty": total_qnty}