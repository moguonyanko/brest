from fastapi import FastAPI

app = FastAPI(
    title="Brest GIS API",
    description="GISの計算機能をREST APIで提供する。",
    summary="Brest GIS API by REST",
    version="0.0.1"
)

def get_coordinates(feature, index: int = 0) -> list:
    if len(feature) == 0:
        return []
    if "geometry" in feature:
        return feature["geometry"]["coordinates"]
    else:
        return feature["features"][index]["geometry"]["coordinates"]

def point_on_which_side(px, py, lx1, ly1, lx2, ly2) -> int:
    cp = (px - lx1) * (ly2 - ly1) - (py - ly1) * (lx2 - lx1)
    if cp > 0: #right
        return 1 
    elif cp < 0: #left
        return -1
    else: #on the line
        return 0

#GeoJSONの形式でリクエストしたいのでPOSTにしている。
@app.post("/pointsideofline/", tags=["geometry"])
async def get_point_side_of_line(line: dict, point: dict):
    lps = get_coordinates(line)
    pps = get_coordinates(point)
    side = point_on_which_side(pps[1], pps[0], lps[0][1], lps[0][0], lps[1][1], lps[1][0])
    return {"side": side}

@app.get("/hellogis/", tags=["test"])
async def request_ping():
    return { "message": "Hello Brest GIS!" }
