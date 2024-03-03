from fastapi import FastAPI

app = FastAPI(
    title="My GIS API",
    description="GISの計算機能をREST APIで提供する。",
    summary="GIS REST API",
    version="0.0.1"
)

def get_coordinates(feature, index: int = 0) -> list:
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

@app.get("/hellomygis/", tags=["test"])
async def request_ping():
    return { "message": "Hello My GIS!" }
