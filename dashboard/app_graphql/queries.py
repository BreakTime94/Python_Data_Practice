from typing import List, Optional

import pandas as pd
from sqlalchemy import text
import strawberry

from api.db.db_conn import engine
from .types import SalesViewRow


@strawberry.type
class Query:
    @strawberry.field
    def sales_view_all(
        self,
        limit: int = 20000,
    ) -> List[SalesViewRow]:
        """
        전체 데이터 조회용 Query
        """
        sql = """
            SELECT
                date_id,
                year,
                quarter,
                month_no,
                month_name,
                customer_name,
                gender,
                birth_date,
                age,
                product_name,
                color,
                product_category_name,
                category_name,
                sido,
                sigungu,
                region,
                channel_name,
                promotion_name,
                discount_rate,
                quantity,
                sales_unit_price,
                sales_amount,
                cost_price,
                cost_amount,
                net_profit
            FROM sales_view_table
            ORDER BY date_id
            LIMIT :limit
        """

        df = pd.read_sql_query(text(sql), con=engine, params={"limit": limit})
        records = df.to_dict(orient="records")

        return [SalesViewRow(**row) for row in records]