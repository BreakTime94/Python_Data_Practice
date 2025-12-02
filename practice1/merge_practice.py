import pandas as pd
from fontTools.varLib.cff import merge_region_fonts

if __name__ == "__main__":
  sales = pd.read_excel("data/Sales.xlsx", sheet_name="Sheet1")
  details = pd.read_excel("data/Details.xlsx", sheet_name=None)
  # 엑셀에 적혀져 있는  sheet 명으로 가져옴
  promotion = details['프로모션']
  channel = details['채널']
  region = details['지역']
  category = details['분류']
  product_category = details['제품분류']
  product = details['제품']
  date = details['날짜']
  date = pd.to_datetime(date["날짜"])
  customer = details["2018년도~2022년도 주문고객"]

  print(date.keys)

  merge_df = pd.merge(sales, date, on="날짜", how="left") # 왼쪽 오른쪽 테이블의 명이 다를 때는 left_on을 쓴다.
  merge_df = pd.merge(merge_df, product, on="제품코드", how="left")
  merge_df = pd.merge(merge_df, customer, on="고객코드", how="left")
  merge_df = pd.merge(merge_df, promotion, on="프로모션코드", how="left")
  merge_df = pd.merge(merge_df, channel, on ="채널코드", how="left")
  # 제품 및 분류를 넣을 때는 계층구조가 중요하다. 가장 하위 카테고리인 제품을 먼저 붙이고 그 다음 상위 카테고리를 붙여야 외래키 오류가 나지 않는다.
  merge_df = pd.merge(merge_df, product_category, on="제품분류코드", how="left")
  merge_df = pd.merge(merge_df, category, on="분류코드", how="left")
  merge_df = pd.merge(merge_df, region, on="지역코드", how="left")
  print(merge_df.keys())

  merge_df = merge_df [['날짜','고객명', 'Quantity', '단가', '원가', '지역_x',
       '제품명', '색상', '프로모션', '할인율', '채널명', '제품분류명', '분류명', '시도', '구군시']]

  merge_df.rename({"Quantity": "수량", "지역_x": "지역"}, axis=1, inplace=True)

  merge_df["판매량"] = merge_df["수량"] * ((merge_df["단가"])* (1 - merge_df["할인율"]) - merge_df["원가"])

  product_group_revenue = merge_df.groupby("제품명")["판매량"].sum().sort_values(ascending=False)

  print(product_group_revenue)

