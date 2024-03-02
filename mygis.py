'''
参考:
https://fastapi.tiangolo.com/tutorial/bigger-applications/
'''
from fastapi import FastAPI

app = FastAPI(
    title="My GIS API",
    description="GISに関連する計算処理をRESTによるAPIで提供する。",
    summary="GIS REST API",
    version="0.0.1"
)

@app.get("/hello/", tags=["test"])
async def request_ping():
    return { "message": "Hello My GIS API!" }

#GeoJSONの形式でリクエストしたいのでPOSTにしている。
@app.post("/pointsideofline/", tags=["geometry"])
async def get_point_side_of_line(line: dict, point: dict):
    return { "message": "Implement now" }

