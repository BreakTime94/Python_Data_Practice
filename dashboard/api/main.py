from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from fastapi.middleware.wsgi import WSGIMiddleware
from app_graphql.schema import schema
from api.app_dash.dashboard import flask_app

app = FastAPI(
  title="Sales View GraphQL, API",
  version="1.0",
)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/app_graphql")

# flask dash board 연결
app.mount("/dashboard", WSGIMiddleware(flask_app))


@app.get("/")
def read_root():
  return {
    "message": "api 호출이라능~ FastAPI, GraphQL, flask/Dash 라이브러리 썼다능",
    "graphql": "/app_graphql",
    "dashboard": "/dashboard"
  }