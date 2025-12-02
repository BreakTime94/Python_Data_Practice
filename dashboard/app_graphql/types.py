import datetime
from typing import Optional
import strawberry

# null 값 허용 -> Optional
@strawberry.type
class SalesViewRow:
    # 날짜 관련
    date_id: datetime.date
    year: int
    quarter: int
    month_no: int
    month_name: str

    # 고객
    customer_name: str
    gender: Optional[str]
    birth_date: Optional[datetime.date]
    age: Optional[float]

    # 제품
    product_name: str
    color: Optional[str]

    # 제품분류
    product_category_name: Optional[str]

    # 🔥 상위 분류 추가
    category_name: Optional[str]

    # 지역
    sido: str
    sigungu: str
    region: str

    # 채널
    channel_name: str

    # 프로모션
    promotion_name: Optional[str]
    discount_rate: Optional[float]

    # 매출/원가/이익
    quantity: int
    sales_unit_price: float
    sales_amount: float
    cost_price: float
    cost_amount: float
    net_profit: float