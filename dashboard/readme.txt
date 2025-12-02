python -m uvicorn api.main:app --reload

query {
  salesViewAll(limit: 1000) {
    dateId
    year
    quarter
    monthNo
    monthName
    customerName
    gender
    birthDate
    age
    productName
    color
    productCategoryName
    sido
    sigungu
    region
    channelName
    promotionName
    discountRate
    quantity
    salesUnitPrice
    salesAmount
    costPrice
    costAmount
    netProfit
  }
}