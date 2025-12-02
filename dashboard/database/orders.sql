-- =========================================================
-- 0. DROP TABLES (FK 순서 고려)
-- =========================================================
DROP TABLE IF EXISTS fact_sales CASCADE;

DROP TABLE IF EXISTS dim_product CASCADE;
DROP TABLE IF EXISTS dim_product_category CASCADE;
DROP TABLE IF EXISTS dim_category CASCADE;

DROP TABLE IF EXISTS dim_customer CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_channel CASCADE;
DROP TABLE IF EXISTS dim_promotion CASCADE;
DROP TABLE IF EXISTS dim_region CASCADE;


-- =========================================================
-- 1) 지역
-- =========================================================
CREATE TABLE dim_region (
    region_code   INTEGER PRIMARY KEY,
    sido          VARCHAR(50),
    sigungu       VARCHAR(50),
    region        VARCHAR(100)
);

-- =========================================================
-- 2) 프로모션
-- =========================================================
CREATE TABLE dim_promotion (
    promotion_code INTEGER PRIMARY KEY,
    promotion_name VARCHAR(100),
    discount_rate  NUMERIC(5, 2)
);

-- =========================================================
-- 3) 채널
-- =========================================================
CREATE TABLE dim_channel (
    channel_code INTEGER PRIMARY KEY,
    channel_name VARCHAR(100)
);

-- =========================================================
-- 4) 날짜
-- =========================================================
CREATE TABLE dim_date (
    date_id        DATE PRIMARY KEY,
    year           INTEGER NOT NULL,
    quarter        INTEGER NOT NULL,
    month_no       INTEGER NOT NULL,
    month_name     VARCHAR(20) NOT NULL
);

-- =========================================================
-- 5) 상위 분류 (카테고리)  : dim_category
--    예: "가전", "의류", "생활용품" 등
-- =========================================================
CREATE TABLE dim_category (
    category_id   SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL
);

-- =========================================================
-- 6) 제품분류 (중분류) : dim_product_category
--    예: "TV", "냉장고", "셔츠", "바지" 등
-- =========================================================
CREATE TABLE dim_product_category (
    product_category_id   SERIAL PRIMARY KEY,
    product_category_code VARCHAR(20) UNIQUE,     -- 엑셀의 '제품분류코드'(E1, E2, ...)
    product_category_name VARCHAR(100) NOT NULL,  -- 엑셀의 '제품분류명'
    category_id           INTEGER NOT NULL REFERENCES dim_category(category_id)
);

-- =========================================================
-- 7) 제품 : dim_product
-- =========================================================
CREATE TABLE dim_product (
    product_code        INTEGER PRIMARY KEY,
    product_name        VARCHAR(200),
    color               VARCHAR(50),
    cost_price          NUMERIC(18,2),
    unit_price          NUMERIC(18,2),
    product_category_id INTEGER REFERENCES dim_product_category(product_category_id)
);

-- =========================================================
-- 8) 주문고객 : dim_customer
-- =========================================================
CREATE TABLE dim_customer (
    customer_code  INTEGER PRIMARY KEY,
    region_code    INTEGER REFERENCES dim_region(region_code),
    customer_name  VARCHAR(100),
    gender         VARCHAR(10),
    birth_date     DATE
);

-- =========================================================
-- 9) 매출 팩트 : fact_sales
-- =========================================================
CREATE TABLE fact_sales (
    sales_id       SERIAL PRIMARY KEY,
    date_id        DATE    NOT NULL REFERENCES dim_date(date_id),
    product_code   INTEGER NOT NULL REFERENCES dim_product(product_code),
    customer_code  INTEGER NOT NULL REFERENCES dim_customer(customer_code),
    promotion_code INTEGER NOT NULL REFERENCES dim_promotion(promotion_code),
    channel_code   INTEGER NOT NULL REFERENCES dim_channel(channel_code),
    region_code    INTEGER NOT NULL REFERENCES dim_region(region_code),

    quantity       INTEGER       NOT NULL,
    unit_price     NUMERIC(18,2) NOT NULL,
    total_price    NUMERIC(18,2) NOT NULL
);
