'''
参考:
https://fastapi.tiangolo.com/tutorial/bigger-applications/
'''
from fastapi import FastAPI

app = FastAPI(
    title="My Mapping API",
    description="GISに関連する計算処理をRESTによるAPIで提供する。",
    summary="GIS REST API",
    version="0.0.1"
)

@app.get("/ping/", tags=["test"])
async def request_ping():
    return { "message": "pong" }
