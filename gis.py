from typing import Union
import json
import math

from fastapi import FastAPI, HTTPException, status

from shapely import *
from shapely.ops import triangulate, voronoi_diagram, split

app = FastAPI(
    title="Brest GIS API",
    description="GISの計算機能をREST APIで提供する。現時点では2次元のみの対応とする。",
    summary="Brest GIS API by REST",
    version="0.0.1"
)

@app.get("/hellogis/", tags=["test"])
async def get_hellogis():
    return { "message": "Hello Brest GIS!" }

def get_coordinates(feature, index: int = 0) -> list:
    if len(feature) == 0:
        return []
    if "geometry" in feature:
        return feature["geometry"]["coordinates"]
    else:
        return feature["features"][index]["geometry"]["coordinates"]

def get_gradient(lx1: Union[int, float], ly1: Union[int, float], 
                 lx2: Union[int, float], ly2: Union[int, float]) -> Union[int, float]:
    dy = ly2 - ly1
    dx = lx2 - lx1
    gradient = dy / dx
    return gradient

def get_intercept(lx1: Union[int, float], ly1: Union[int, float], 
                  gradient: Union[int, float]) -> Union[int, float]:
    return ly1 - gradient * lx1

'''
点の座標が方程式を満たしていたら点が直線上にあると判定する。
'''
def is_on_line(px, py, lx1, ly1, lx2, ly2) -> bool:
    gradient = get_gradient(lx1, ly1, lx2, ly2)
    intercept = get_intercept(lx1, ly1, gradient)

    # 点Pが方程式を満たしているかどうか
    if gradient == 0:
        is_on_line = px == lx1
    elif intercept == 0:
        is_on_line = py == gradient * px
    else:
        is_on_line = py == gradient * px + intercept
    
    return is_on_line

'''
ベクトルの外積から引数の点が線分を挟んでどちら側にあるのかを求める。
'''
def point_on_which_side(px, py, lx1, ly1, lx2, ly2) -> int:
    cp = (px - lx1) * (ly2 - ly1) - (py - ly1) * (lx2 - lx1)
    if cp > 0: #right
        return 1 
    elif cp < 0: #left
        return -1
    else: #on the line #既にis_on_line()で判定しているのでここには到達しない。
        return 0

#GeoJSONの形式でリクエストしたいのでPOSTにしている。
@app.post("/pointsideofline/", tags=["geometry"])
async def get_point_side_of_line(line: dict, point: dict):
    lps = get_coordinates(line)
    pps = get_coordinates(point)
    if len(lps) == 0 or len(pps) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, 
                            detail="点と線分の座標は必須です。")
    px = pps[1]
    py = pps[0]
    lx1 = lps[0][1]
    ly1 = lps[0][0] 
    lx2 = lps[1][1] 
    ly2 = lps[1][0]
    if is_on_line(px, py, lx1, ly1, lx2, ly2):
        return {"side": 0}
    else:
        side = point_on_which_side(px, py, lx1, ly1, lx2, ly2)
        return {"side": side}
    
class Vector():
    def __init__(self, x = 0, y = 0, z = 0):
        self.x = x
        self.y = y
        self.z = z

def cross_product_z(vec1: Vector, vec2: Vector):
    return vec1.x * vec2.y - vec1.y * vec2.x

#可読性のためVectorとはとりあえず分けて定義している。
class Point:
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y

    def __iter__(self):
        return iter((self.x, self.y))
        # yield self.x
        # yield self.y

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def __iter__(self):
        return iter((self.x1, self.y, self.x2, self.y2))
        # yield self.x1
        # yield self.y1
        # yield self.x2
        # yield self.y2

    def get_vector(self) -> Vector:
        return Vector(self.x2 - self.x1, self.y2 - self.y1)

def is_in_range(line: Line, point: Point):
    # x1, y1, x2, y2 = *line
    # x, y = *point
    x1 = line.x1
    y1 = line.y1
    x2 = line.x2
    y2 = line.y2
    x = point.x
    y = point.y
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

def is_parallel_or_overlap(line1: Line, line2: Line):
    pass

@app.post("/linecrosscheck/", tags=["geometry"], response_model=dict[str, bool])
async def check_cross_lines(line1: dict, line2: dict):
    lps1 = get_coordinates(line1)
    lps2 = get_coordinates(line2)

    line1 = LineString(lps1)
    line2 = LineString(lps2)

    response = {
        "result": line1.intersects(line2)
    }

    return response

@app.post("/convexhull/", tags=["geometry"])
async def calc_convex_hull(multipoint: dict):
    geom = multipoint["features"][0]["geometry"]
    mp = from_geojson(json.dumps(geom))
    polygon = convex_hull(mp)
    #json.loadsを介さないとブラウザ側でJSON.parseを行う必要が生じる。
    return {"result": json.loads(to_geojson(polygon))}

@app.post("/triangulation/", tags=["geometry"])
async def calc_trianglation(geom: dict):
    response = {"result": {}}
    if len(geom["features"]) == 0:
        return response
    g = geom["features"][0]["geometry"]
    primitive = from_geojson(json.dumps(g))
    result = triangulate(primitive)
    geojsons = to_geojson(result)
    result = [json.loads(geojson) for geojson in geojsons]
    response["result"] = result
    return response

@app.post("/minimumboundingcircle/", tags=["geometry"])
async def calc_minimum_bounding_circle(geojson: dict):
    points = [from_geojson(json.dumps(feature["geometry"])) 
              for feature in geojson["features"]]
    mp = MultiPoint(points)

    #凸包を使って求める場合
    # ch = convex_hull(mp)
    # circle = minimum_bounding_circle(ch)

    #最小回転矩形を使って求める場合
    rotated_rectangle = oriented_envelope(mp)
    center = rotated_rectangle.centroid
    bounds = rotated_rectangle.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    radius = math.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    circle = buffer(center, radius)

    return {"result": json.loads(to_geojson(circle))}

def get_geometris_from_geojson(geojson: dict):
    return [from_geojson(json.dumps(feature["geometry"])) 
            for feature in geojson["features"]]   

def get_geojson_from_geometry(geometry):
     return json.loads(to_geojson(geometry))

@app.post("/contains/", tags=["geometry"])
async def calc_contains(area_geojson: dict, target_geojson: dict):
    areas = get_geometris_from_geojson(area_geojson)
    targets = get_geometris_from_geojson(target_geojson)

    result = {}

    for index, area in enumerate(areas):
        result[index] = []
        for target in targets:            
            if contains(area, target):
                result[index].append(get_geojson_from_geometry(target))

    return { "result": result }

@app.post("/voronoidiagram/", tags=["geometry"])
async def calc_voronoi_diagram(points: dict):
    ps = get_geometris_from_geojson(points)
    mp = MultiPoint(ps)
    result = voronoi_diagram(mp)

    return { "result": get_geojson_from_geometry(result) }

@app.post("/splitpolygon/", tags=["geometry"])
async def split_polygon_by_line(polygon: dict, line: dict):
    if len(polygon) == 0 or len(line) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid geometry")
    polygon_geom = get_geometris_from_geojson(polygon)
    line_geom = get_geometris_from_geojson(line)
    tmp = polygon_geom[0]
    for splitter in line_geom:
        tmp = MultiPolygon(split(tmp, splitter)) #分割のためだけにMultiPolygonにする。
    result = GeometryCollection(tmp.geoms)

    return { "result": get_geojson_from_geometry(result) }
