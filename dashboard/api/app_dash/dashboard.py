import flask
from dash import Dash, html, dcc, Input, Output
from .graphql_client import fetch_sales_view_all

flask_app = flask.Flask(__name__)

dash_app = Dash(
  __name__,
  server=flask_app,
  requests_pathname_prefix='/dashboard/',
  suppress_callback_exceptions=True,
)

def card_style():
  return {
    "flex": "1",
    "margin": "0 10px",
    "padding": "10px",
    "backgroundColor": "#F8F9FA",
    "borderRadius":"5px",
    "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.15)",
    "textAlign": "center",
  }

dash_app.layout=html.Div(
  style={"padding": "5px"},
  children=[
    html.H2("매출분석 대시보드", style={"textAlign": "center"}),
      html.Div(
       style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"},
        children=[
          html.Div(style=card_style(), id="total_sales", children=[html.H4("총 매출액")]),
          html.Div(style=card_style(), id="total_profit", children=[html.H4("순이익")]),
          html.Div(style=card_style(), id="total_customer", children=[html.H4("총 고객수")]),
          html.Div(style=card_style(), id="total_qnty", children=[html.H4("총 판매량")]),
        ]
      )
  ]
)


# Output : return하는 값, Input: parameter
@dash_app.callback(
  [
    Output("total_sales", "children"),
    Output("total_profit", "children"),
    Output("total_customer", "children"),
    Output("total_qnty", "children"),
  ],
  Input("total_sales", "value"),

)
def update_dashboard(value):
  df = fetch_sales_view_all()
  return (
    [html.H2("총 매출액"), html.H2("1200원")],
    [html.H2("총 매출액"), html.H2("1200원")],
    [html.H2("총 매출액"), html.H2("1200원")],
    [html.H2("총 매출액"), html.H2("1200원")]
  )