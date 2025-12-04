import calendar

import flask
from dash import Dash, html, dcc, Input, Output
from plotly.express import treemap

from .graphql_client import fetch_sales_view_all
from .modules.cards import bring_card_data
from .modules.category_product_treemap import category_product_treemap_graph
from .modules.region_sigungu_bar import region_sigungu_bar_graph
from .modules.year_bar import year_bar_graph
from .modules.year_month_line import year_month_graph

flask_app = flask.Flask(__name__)

dash_app = Dash(
    __name__,
    server=flask_app,
    requests_pathname_prefix="/dashboard/",
    suppress_callback_exceptions=True,
)

def card_style():
    return {
        "flex": "1",
        "margin": "0 10px",
        "padding": "20px",
        "backgroundColor": "#F8F9FA",
        "borderRadius": "10px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.15)",
        "textAlign": "center",
    }


dash_app.layout = (
    html.Div(
    style={"padding": "20px"},
    children=[
        html.H2("매출 분석 대시보드", style={"textAlign": "center"}),
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": "30px",
            },
            children=[
                html.Div(id="card_total_sales", style=card_style(),  children =[html.H4('총매출액')]),
                html.Div(id="card_total_profit", style=card_style(), children =[html.H4('전체 순이익')]),
                html.Div(id="card_total_customers", style=card_style(),children =[html.H4('총 고객수')]),
                html.Div(id="card_total_qnty", style=card_style(), children =[html.H4('총 판매수량')])
            ]
        ),
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": "30px",
            },
            children=[
                html.Div(
                    dcc.Graph(id="chart-year-bar"),
                    style={"flex": "1", "height": "380px"},
                ),
                html.Div(
                    dcc.Graph(id="chart-year-line"),
                    style={"flex": "1", "height": "380px"},
                ),
            ]
        ),

      html.Div(
        style={
          "display": "flex",
          "justifyContent": "space-between",
          "gap": "20px",
          "marginBottom": "30px",
        },
        children=[
          # ===================
          #   지역 필터 + 지역 그래프
          # ===================
          html.Div(
            style={"flex": "1"},
            children=[
              # 지역 필터
              html.Div(
                dcc.Dropdown(
                  id="region-filter",
                  options=[],
                  value="서울",
                  placeholder="지역 선택(미선택 시 전체)",
                  clearable=True,
                  style={"width": "60%", "fontSize": "12px"},
                ),
                style={"marginBottom": "15px"},
              ),
              # 시군구 차트
              dcc.Graph(id="region-sigungu-chart", style={"width": "100%", "height": "450px"}),
            ],
          ),

          # ===================
          #   카테고리 필터 + 트리맵 그래프
          # ===================
          html.Div(
            style={"flex": "1"},
            children=[
              # 카테고리 필터
              html.Div(
                dcc.Dropdown(
                  id="category-filter",
                  options=[
                    {"label": "제품", "value": "productName"},
                    {"label": "제품분류", "value": "productCategoryName"},
                    {"label": "대분류", "value": "categoryName"},
                  ],
                  value="categoryName",
                  clearable=True,
                  placeholder="categoryName",
                  style={"width": "60%", "fontSize": "12px"},
                ),
                style={"marginBottom": "15px"},
              ),
              # 트리맵
              dcc.Graph(id="category-treemap", style={"width": "100%", "height": "450px"}),
            ],
          ),
        ],
      )

    ]
))
import plotly.express as px

# Output: return, Input: parameter
@dash_app.callback(
    [
        Output("card_total_sales", "children"),
        Output("card_total_profit", "children"),
        Output("card_total_customers", "children"),
        Output("card_total_qnty", "children"),
        Output("chart-year-bar", "figure"),
        Output("chart-year-line", "figure"),
        Output("region-filter", "options"),
        Output("region-sigungu-chart", "figure"),
        Output("category-filter", "options"),
        Output("category-treemap", "figure"),
    ],
    [Input("region-filter", "value"), Input("category-filter", "value")],

)
def update_dashboard(selected_region, selected_category):
    df = fetch_sales_view_all()

    region_options, fig_bar_sigungu = region_sigungu_bar_graph(df, selected_region)

    fig_bar_year = year_bar_graph(df)
    fig_line_year = year_month_graph(df)
    cardData = bring_card_data(df)

    treemap_options, fig_treemap = category_product_treemap_graph(df, selected_category)


    return (
        [html.H4("총매출액"), html.H2(f"{cardData['total_sales']:,}원")],
        [html.H4("총매출액"), html.H2(f"{cardData['total_profit']:,}원")],
        [html.H4("총매출액"), html.H2(f"{cardData['total_customers']:,}명")],
        [html.H4("총매출액"), html.H2(f"{cardData['total_qnty']:,}건수")],
        fig_bar_year, fig_line_year, region_options, fig_bar_sigungu, treemap_options, fig_treemap
    )