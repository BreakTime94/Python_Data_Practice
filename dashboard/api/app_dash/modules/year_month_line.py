import plotly.express as px
import calendar


def year_month_graph(df):
  # 월별 매출 (월 순서 정렬 포함)
  month_group = (
    df.groupby(["year", "monthName"], as_index=False)["salesAmount"]
    .sum()
  )

  # ️ Month → Number 매핑 (달력순)
  # 월 이름 → 숫자 매핑
  month_to_num = {month: idx for idx, month in enumerate(calendar.month_name) if month}

  # 숫자 컬럼 추가
  month_group["month_num"] = month_group["monthName"].map(month_to_num)

  # 월 기준 정렬
  month_group = month_group.sort_values(["year", "month_num"])

  # 매핑 후 숫자로 정렬
  month_group["month_num"] = month_group["monthName"].map(month_to_num)

  # month_group = month_group.sort_values("month_num")

  fig_line = px.line(
    month_group,
    x="monthName",
    y="salesAmount",
    color="year",
    markers=True,
    title="연도별 월 매출 추이",
    category_orders={"monthName": list(month_to_num.keys())}
  )

  # 보기 좋게 옵션 추가
  fig_line.update_layout(
    xaxis_title="월",
    yaxis_title="매출액",
    legend_title="연도",
    hovermode="x unified",
  )

  return fig_line