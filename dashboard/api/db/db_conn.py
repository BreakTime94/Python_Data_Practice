from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine

DB_URL = "postgresql+psycopg2://user:1234@localhost:5432/mydb"
engine: Engine = create_engine(DB_URL, echo=False, future=True)

Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)