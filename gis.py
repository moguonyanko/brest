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

#GeoJSONの形式でリクエストしたいのでPOSTにしている。
@app.post("/pointsideofline/", tags=["geometry"])
async def get_point_side_of_line(line: dict, point: dict):
    line_coordinates = get_coordinates(line)
    point_coordinates = get_coordinates(point)
    return {"line_coordinates": line_coordinates, "point_coordinates": point_coordinates}

@app.get("/hellogis/", tags=["test"])
async def request_ping():
    return { "message": "Hello Brest GIS!" }
