'''
参考:
https://fastapi.tiangolo.com/tutorial/sql-databases/#create-the-database-models
'''
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "mysql://sampleuser:samplepass@localhost:3306/test"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args = {
      "zeroDateTimeBehavior": "CONVERT_TO_NUL"
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
