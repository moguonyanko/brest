import psycopg
from typing import Union, Annotated, Any
from fastapi import FastAPI, HTTPException, status, Body, Depends
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

app = FastAPI(
    title="Brest Database API",
    description="Databaseを操作する機能をAPIで提供する。",
    summary="BrestDatabase API by REST",
    version="0.0.1"
)

#TODO: psycopg2がインストールできないとcreate_engineでエラーが発生する。
def get_db():
    connect_url = 'postgresql://postgres:postgres@localhost:5432/postgres'
    connect_args = {"check_same_thread": False}
    engine = create_engine(connect_url, connect_args=connect_args)
    with Session(engine) as session:
        yield session

class SqlRequest(BaseModel):
    sql: str

'''
引数のクエリを実行して結果を返す。
FastAPIの機能を使わずpsycopgだけでデータベースを扱っている。
'''
def execute_query(query: str) -> list:
    con_str = "postgresql://postgres:postgres@localhost:5432/postgres"
    with psycopg.connect(con_str, autocommit=True) as conn:
        with conn.cursor() as cur:    
            cur.execute(query)
            return cur.fetchall()

@app.post("/injectsql/", tags=["database"], response_model=dict[str, Any])
async def inject_sql(sql_request: SqlRequest):
    results = execute_query(sql_request.sql)
    return {
        "results": results
    }
    
'''
データベースの状態は変化するのでカーディナリティや選択率は同じ条件で要求した場合でも
異なる結果を返す可能性がある。すなわち冪等ではない。しかるにgetではなくpostにしている。
'''    
@app.post("/cardinarity/", tags=["database"], response_model=dict[str, Any])
async def get_cardinarity(table_info: dict):
    table_name = table_info['table']
    column_names = table_info['columns']
    sql = f'SELECT COUNT(DISTINCT {','.join(column_names)}) FROM {table_name}'
    results = execute_query(sql)
    return {
        "results": results
    }

@app.post("/selectivity/", tags=["database"], response_model=dict[str, Any])
async def get_selectivit(table_info: dict):
    table_name = table_info['table']
    condition = table_info['condition']
    sql = f'SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name}) AS selection_rate FROM {table_name} WHERE {condition};'
    results = execute_query(sql)
    return {
        "results": results
    }
