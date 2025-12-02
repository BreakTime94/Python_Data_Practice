import os, pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg2://user:1234@localhost:5432/mydb"
engine = create_engine(DB_URL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)

details_path = os.path.join(BASE_DIR, "database", "Details.xlsx")
sales_path = os.path.join(BASE_DIR, "database", "Sales.xlsx")
orders_sql_path = os.path.join(BASE_DIR, "database", "orders.sql")

def run_sql_file(engine, file_path):
  with open(file_path, 'r', encoding="utf-8") as f:
    sql = f.read()
  with engine.begin() as conn:
    conn.execute(text(sql))

run_sql_file(engine, orders_sql_path)


# ===========================================
# 3. 엑셀 데이터 읽기
# ===========================================
SHEET_NAMES = {
    "region": "지역",
    "channel": "채널",
    "customer": "2018년도~2022년도 주문고객",
    "product": "제품",
    "product_category": "제품분류",
    "category": "분류",          # 🔥 상위 분류 시트
    "promotion": "프로모션",
}

details_all = pd.read_excel(details_path, sheet_name=None, engine="openpyxl")
sales_df = pd.read_excel(sales_path, engine="openpyxl")

region_df = details_all[SHEET_NAMES["region"]]
channel_df = details_all[SHEET_NAMES["channel"]]
customer_df = details_all[SHEET_NAMES["customer"]]
product_df = details_all[SHEET_NAMES["product"]]
product_category_df = details_all[SHEET_NAMES["product_category"]]
category_df = details_all[SHEET_NAMES["category"]]    # 🔥 신규
promotion_df = details_all[SHEET_NAMES["promotion"]]

print("✅ Details.xlsx & Sales.xlsx loaded")


# ===========================================
# 4. 컬럼명 정리
# ===========================================

# 4-1) 지역 (dim_region)
region_df = region_df.rename(
    columns={
        "지역코드": "region_code",
        "시도": "sido",
        "구군시": "sigungu",
        "지역": "region",
    }
)

# 4-2) 채널 (dim_channel)
channel_df = channel_df.rename(
    columns={
        "채널코드": "channel_code",
        "채널명": "channel_name",
    }
)

# 4-3) 주문고객 (dim_customer)
customer_df = customer_df.rename(
    columns={
        "고객코드": "customer_code",
        "지역코드": "region_code",
        "고객명": "customer_name",
        "성별": "gender",
        "생년월일": "birth_date",
    }
)
customer_df["birth_date"] = pd.to_datetime(
    customer_df["birth_date"], errors="coerce"
).dt.date

# 4-4) 제품 (dim_product 원본)
product_df = product_df.rename(
    columns={
        "제품코드": "product_code",
        "제품명": "product_name",
        "색상": "color",
        "원가": "cost_price",
        "단가": "unit_price",
        "제품분류코드": "product_category_code",
        "제품분퓨코드": "product_category_code",  # 혹시 오타 대비
    }
)

# 4-5) 제품분류 (중분류 시트: 제품분류코드, 제품분류명, 분류코드)
product_category_df = product_category_df.rename(
    columns={
        "제품분류코드": "product_category_code",
        "제품분류명": "product_category_name",
        "분류코드": "category_id",   # 🔥 상위분류 코드 (1,2,3,...)
    }
)

# 4-6) 상위 분류 (분류 시트: 분류코드, 분류명)
category_df = category_df.rename(
    columns={
        "분류코드": "category_id",
        "분류명": "category_name",
    }
)

print("📌 category_df.head():")
print(category_df.head())

# 4-7) 프로모션
promotion_df = promotion_df.rename(
    columns={
        "프로모션코드": "promotion_code",
        "프로모션": "promotion_name",
        "할인율": "discount_rate",
    }
)

# 4-8) Sales (fact_sales용)
sales_df = sales_df.rename(
    columns={
        "날짜": "date_id",
        "제품코드": "product_code",
        "고객코드": "customer_code",
        "프로모션코드": "promotion_code",
        "채널코드": "channel_code",
        "Quantity": "quantity",
        "UnitPrice": "unit_price",
    }
)
sales_df["date_id"] = pd.to_datetime(sales_df["date_id"], errors="coerce").dt.date
sales_df["quantity"] = (
    pd.to_numeric(sales_df["quantity"], errors="coerce").fillna(0).astype(int)
)
sales_df["unit_price"] = (
    pd.to_numeric(sales_df["unit_price"], errors="coerce").fillna(0).astype(float)
)

print("✅ Column rename & type casting done")


# ===========================================
# 5. DIM 테이블 적재
# ===========================================

# 5-1) dim_region
dim_region = region_df.drop_duplicates(subset=["region_code"])
dim_region.to_sql("dim_region", engine, if_exists="append", index=False)
print("➡️ dim_region inserted")

# 5-2) dim_channel
dim_channel = channel_df.drop_duplicates(subset=["channel_code"])
dim_channel.to_sql("dim_channel", engine, if_exists="append", index=False)
print("➡️ dim_channel inserted")

# 5-3) dim_promotion
dim_promotion = promotion_df.drop_duplicates(subset=["promotion_code"])
dim_promotion.to_sql("dim_promotion", engine, if_exists="append", index=False)
print("➡️ dim_promotion inserted")

# 5-4) dim_category (상위 분류)  ← 분류 시트 그대로 사용
dim_category = category_df[["category_id", "category_name"]].drop_duplicates()
dim_category.to_sql("dim_category", engine, if_exists="append", index=False)
print("➡️ dim_category inserted")
print("   dim_category.head():")
print(dim_category.head())

# 5-5) dim_customer
dim_customer = customer_df[
    ["customer_code", "region_code", "customer_name", "gender", "birth_date"]
].drop_duplicates(subset=["customer_code"])
dim_customer.to_sql("dim_customer", engine, if_exists="append", index=False)
print("➡️ dim_customer inserted")

# 5-6) dim_product_category (중분류)
#   - 제품분류코드(E1,E2...), 제품분류명, category_id (상위 분류 FK)
dim_product_category = product_category_df[
    ["product_category_code", "product_category_name", "category_id"]
].drop_duplicates(subset=["product_category_code"])

# SERIAL PK용 product_category_id 생성 (코드에서 직접 부여해도 무방)
dim_product_category = dim_product_category.sort_values("product_category_code").reset_index(drop=True)
dim_product_category["product_category_id"] = range(1, len(dim_product_category) + 1)

dim_product_category_db = dim_product_category[
    ["product_category_id", "product_category_code", "product_category_name", "category_id"]
]
dim_product_category_db.to_sql("dim_product_category", engine, if_exists="append", index=False)
print("➡️ dim_product_category inserted")
print("   dim_product_category.head():")
print(dim_product_category_db.head())

# 5-7) dim_product (제품 + 중분류 FK)
dim_product_base = product_df[
    [
        "product_code",
        "product_name",
        "color",
        "cost_price",
        "unit_price",
        "product_category_code",
    ]
].drop_duplicates(subset=["product_code"])

dim_product = dim_product_base.merge(
    dim_product_category_db[["product_category_id", "product_category_code"]],
    on="product_category_code",
    how="left",
)

dim_product = dim_product[
    [
        "product_code",
        "product_name",
        "color",
        "cost_price",
        "unit_price",
        "product_category_id",
    ]
]
dim_product.to_sql("dim_product", engine, if_exists="append", index=False)
print("➡️ dim_product inserted")
print("   dim_product.head():")
print(dim_product.head())

# 5-8) dim_date  (Sales의 date_id 기준 생성)
unique_dates = sales_df["date_id"].dropna().drop_duplicates().sort_values()
dim_date = pd.DataFrame({"date_id": unique_dates})

dim_date["year"] = dim_date["date_id"].apply(lambda d: d.year)
dim_date["quarter"] = dim_date["date_id"].apply(lambda d: (d.month - 1) // 3 + 1)
dim_date["month_no"] = dim_date["date_id"].apply(lambda d: d.month)
dim_date["month_name"] = dim_date["date_id"].apply(lambda d: d.strftime("%B"))

dim_date.to_sql("dim_date", engine, if_exists="append", index=False)
print("➡️ dim_date inserted")


# ===========================================
# 6. fact_sales 적재
# ===========================================
sales_with_region = sales_df.merge(
    dim_customer[["customer_code", "region_code"]],
    on="customer_code",
    how="left",
)

sales_with_region["total_price"] = (
    sales_with_region["quantity"] * sales_with_region["unit_price"]
)

fact_sales = sales_with_region[
    [
        "date_id",
        "product_code",
        "customer_code",
        "promotion_code",
        "channel_code",
        "region_code",
        "quantity",
        "unit_price",
        "total_price",
    ]
]

fact_sales.to_sql("fact_sales", engine, if_exists="append", index=False)
print("➡️ fact_sales inserted")


# ===========================================
# 7. 뷰 생성: sales_view_table
# ===========================================
VIEW_SQL = """
CREATE OR REPLACE VIEW sales_view_table AS
SELECT
    -- 날짜 관련
    fs.date_id,
    dd.year,
    dd.quarter,
    dd.month_no,
    dd.month_name,

    -- 고객
    dc.customer_name,
    dc.gender,
    dc.birth_date,
    CASE
        WHEN dc.birth_date IS NOT NULL THEN
            EXTRACT(YEAR FROM age(fs.date_id::timestamp, dc.birth_date::timestamp))
        ELSE NULL
    END AS age,

    -- 제품 및 계층
    dp.product_name,
    dp.color,
    dpc.product_category_name,
    dcat.category_name,

    -- 지역 관련
    dr.sido,
    dr.sigungu,
    dr.region,

    -- 채널
    ch.channel_name,

    -- 프로모션
    pr.promotion_name,
    pr.discount_rate,

    -- 매출 / 원가 / 이익
    fs.quantity,
    fs.unit_price               AS sales_unit_price,
    fs.total_price              AS sales_amount,
    COALESCE(dp.cost_price, 0)  AS cost_price,
    fs.quantity * COALESCE(dp.cost_price, 0) AS cost_amount,
    fs.total_price - fs.quantity * COALESCE(dp.cost_price, 0) AS net_profit

FROM fact_sales fs
JOIN dim_date dd
    ON fs.date_id = dd.date_id

JOIN dim_customer dc
    ON fs.customer_code = dc.customer_code

JOIN dim_product dp
    ON fs.product_code = dp.product_code

LEFT JOIN dim_product_category dpc
    ON dp.product_category_id = dpc.product_category_id

LEFT JOIN dim_category dcat
    ON dpc.category_id = dcat.category_id

JOIN dim_region dr
    ON fs.region_code = dr.region_code

JOIN dim_channel ch
    ON fs.channel_code = ch.channel_code

LEFT JOIN dim_promotion pr
    ON fs.promotion_code = pr.promotion_code;
"""

with engine.begin() as conn:
  conn.execute(text(VIEW_SQL))
