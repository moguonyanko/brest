import json
import math
from fastapi import FastAPI, HTTPException, status, Body
from pydantic import BaseModel
from typing import Union, Annotated, Any
from shapely import (
    from_geojson,
    to_geojson,
    oriented_envelope,
    buffer,
    contains,
    convex_hull,
)
from shapely import (
    Polygon,
    LineString,
    MultiPoint,
    GeometryCollection,
    MultiPolygon,
    Point,
)
from shapely import get_x, get_y, MultiLineString, is_valid
from shapely.ops import triangulate, voronoi_diagram, split, nearest_points
from pyproj import Transformer, Geod
from geojsontypes import FeatureCollection
import osmnx as ox
from fastapi_mcp import FastApiMCP
import numpy as np
from k_means_constrained import KMeansConstrained
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="Brest GIS API",
    description="GISの計算機能をREST APIで提供する。現時点では2次元のみの対応とする。",
    summary="Brest GIS API by REST",
    version="0.0.1",
)

mcp = FastApiMCP(
    app,
    name="Practice Fasp-MCP API",
    description="GIS REST APIをAIクライアントで利用できるようにします。",
    describe_full_response_schema=True,
    describe_all_responses=True,
)
mcp.mount()

# But if you re-run the setup, the new endpoints will now be exposed.
mcp.setup_server()


@app.get("/hellogis/", tags=["test"])
async def get_hellogis():
    return {"message": "Hello Brest GIS!"}


@app.get("/helloerror/", tags=["test"])
async def get_hello_error():
    raise HTTPException(
        status_code=400, detail='{"code": "ERR001", "message": "test error"}'
    )


def get_coordinates(feature, index: int = 0) -> list:
    if len(feature) == 0:
        return []
    if "geometry" in feature:
        return feature["geometry"]["coordinates"]
    else:
        return feature["features"][index]["geometry"]["coordinates"]


def get_gradient(
    lx1: Union[int, float],
    ly1: Union[int, float],
    lx2: Union[int, float],
    ly2: Union[int, float],
) -> Union[int, float]:
    dy = ly2 - ly1
    dx = lx2 - lx1
    gradient = dy / dx
    return gradient


def get_intercept(
    lx1: Union[int, float], ly1: Union[int, float], gradient: Union[int, float]
) -> Union[int, float]:
    return ly1 - gradient * lx1


"""
点の座標が方程式を満たしていたら点が直線上にあると判定する。
"""


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


"""
ベクトルの外積から引数の点が線分を挟んでどちら側にあるのかを求める。
"""


def point_on_which_side(px, py, lx1, ly1, lx2, ly2) -> int:
    cp = (px - lx1) * (ly2 - ly1) - (py - ly1) * (lx2 - lx1)
    if cp > 0:  # right
        return 1
    elif cp < 0:  # left
        return -1
    else:  # on the line #既にis_on_line()で判定しているのでここには到達しない。
        return 0


# GeoJSONの形式でリクエストしたいのでPOSTにしている。
@app.post("/pointsideofline/", tags=["geometry"])
async def get_point_side_of_line(line: dict, point: dict):
    lps = get_coordinates(line)
    pps = get_coordinates(point)
    if len(lps) == 0 or len(pps) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="点と線分の座標は必須です。"
        )
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


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


def cross_product_z(vec1: Vector, vec2: Vector):
    return vec1.x * vec2.y - vec1.y * vec2.x


# 可読性のためVectorとはとりあえず分けて定義している。
class MyPoint:
    def __init__(self, x=0, y=0):
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


def is_in_range(line: Line, point: MyPoint):
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

    response = {"result": line1.intersects(line2)}

    return response


@app.post("/convexhull/", tags=["geometry"])
async def calc_convex_hull(multipoint: dict):
    geom = multipoint["features"][0]["geometry"]
    mp = from_geojson(json.dumps(geom))
    polygon = convex_hull(mp)
    # json.loadsを介さないとブラウザ側でJSON.parseを行う必要が生じる。
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
    points = [
        from_geojson(json.dumps(feature["geometry"])) for feature in geojson["features"]
    ]
    mp = MultiPoint(points)

    # 凸包を使って求める場合
    # ch = convex_hull(mp)
    # circle = minimum_bounding_circle(ch)

    # 最小回転矩形を使って求める場合
    rotated_rectangle = oriented_envelope(mp)
    center = rotated_rectangle.centroid
    bounds = rotated_rectangle.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    radius = math.sqrt((width / 2) ** 2 + (height / 2) ** 2)
    circle = buffer(center, radius)

    return {"result": json.loads(to_geojson(circle))}


def get_geometris_from_geojson(geojson: dict):
    return [
        from_geojson(json.dumps(feature["geometry"])) for feature in geojson["features"]
    ]


def is_nested_list(items: list):
    for item in items:
        if isinstance(item, list):
            return True
    return False


def create_geometry(geotype: str, coords: list[float]):
    # 渡されるGeoJSONの形式によっては入れ子になっているので平坦にする。
    if is_nested_list(coords):
        coords = sum(coords, [])

    match geotype:
        case "Point":
            return Point(coords)
        case "LineString":
            return LineString(coords)
        case "Polygon":
            return Polygon(coords)
        case "MultiPolygon":
            return MultiPolygon(coords)
        case "MultiLineString":
            return MultiLineString(coords)
        case "MultiPoint":
            return MultiPoint(coords)
        case _:
            raise TypeError(f"{geotype} is invalid geometry type")


def get_geometris_from_feature_collection(feature_collection: FeatureCollection):
    geoms = []
    for feature in feature_collection.features:
        geotype = feature.geometry.type
        coords = feature.geometry.coordinates
        geom = create_geometry(geotype, coords)
        geoms.append(geom)
    return geoms


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

    return {"result": result}


@app.post("/voronoidiagram/", tags=["geometry"], operation_id="calc_voronoi_diagram")
async def calc_voronoi_diagram(points: dict):
    """
    Voronoi図を計算する。
    """
    ps = get_geometris_from_geojson(points)
    mp = MultiPoint(ps)
    result = voronoi_diagram(mp)

    return {"result": get_geojson_from_geometry(result)}


@app.post("/splitpolygon/", tags=["geometry"], operation_id="split_polygon_by_line")
async def split_polygon_by_line(polygon: dict, line: dict):
    """
    ポリゴンを線分で分割する。

    Args:
        polygon: 分割対象のポリゴン GeoJSON
        例: {"type": "Polygon", "coordinates": [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]}
        line: 分割線 GeoJSON
        例: {"type": "LineString", "coordinates": [[0, 5], [10, 5]]}

    Returns:
        分割後のポリゴン GeoJSON
        例: {"type": "MultiPolygon", "coordinates": [[[[0, 0], [0, 5], [10, 5], [10, 0], [0, 0]]], [[[0, 5], [0, 10], [10, 10], [10, 5], [0, 5]]]]}
    """
    if len(polygon) == 0 or len(line) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid geometry"
        )
    polygon_geom = get_geometris_from_geojson(polygon)
    line_geom = get_geometris_from_geojson(line)
    tmp = polygon_geom[0]
    for splitter in line_geom:
        tmp = MultiPolygon(split(tmp, splitter))  # 分割のためだけにMultiPolygonにする。
    result = GeometryCollection(tmp.geoms)

    return {"result": get_geojson_from_geometry(result)}


@app.post("/nearestpoint/", tags=["geometry"])
async def get_nearest_point(points: dict, line: dict):
    points_geom = get_geometris_from_geojson(points)
    line_geom = get_geometris_from_geojson(line)
    result = nearest_points(line_geom, MultiPoint(points_geom))
    points = [p[0] for p in result]  # resultが入れ子の配列になっている理由は不明

    return {"result": get_geojson_from_geometry(GeometryCollection(points))}


# float型はBodyを使わないとリクエストボディのデータとFastAPIから認識されない。
@app.post("/buffernearpoints/", tags=["geometry"])
async def get_buffer_near_points(
    points: dict, line: dict, distance: Annotated[float, Body()]
):
    points_geom = get_geometris_from_geojson(points)
    line_geom = get_geometris_from_geojson(line)
    buffered_line = buffer(line_geom, distance)
    # filterはイテレータを生成するのでlistを適用してリストに変換する。
    result = list(filter(lambda p: p.intersects(buffered_line), points_geom))

    return {
        "result": get_geojson_from_geometry(GeometryCollection(result)),
        "bufferedLine": get_geojson_from_geometry(buffered_line[0]),
    }


@app.post("/coordconvert/", tags=["geometry"])
async def convert_coords(
    point: dict, fromepsg: Annotated[str, Body()], toepsg: Annotated[str, Body()]
):
    point_geom = get_geometris_from_geojson(point)
    transformer = Transformer.from_crs(
        f"epsg:{fromepsg}", f"epsg:{toepsg}", always_xy=True
    )
    from_x = get_x(point_geom[0])
    from_y = get_y(point_geom[0])
    x, y = transformer.transform(from_x, from_y)
    result_point = Point(x, y)

    return {"result": get_geojson_from_geometry(result_point)}


@app.post(
    "/distance/", tags=["geometry"], response_model=dict[str, Any]
)  # TODO: dictではなく特定のクラスをresponse_modelに指定したい。
async def calc_distance(start: FeatureCollection, goal: FeatureCollection):
    start_geom = get_geometris_from_feature_collection(start)[0]
    goal_geom = get_geometris_from_feature_collection(goal)[0]

    grs80 = Geod(ellps="GRS80")
    start_lat = get_y(start_geom)
    start_lng = get_x(start_geom)
    goal_lat = get_y(goal_geom)
    goal_lng = get_x(goal_geom)

    azimuth, bkw_azimuth, distance = grs80.inv(start_lng, start_lat, goal_lng, goal_lat)
    line = LineString([start_geom, goal_geom])
    return {
        "distance": distance,
        "azimuth": azimuth,
        "bkw_azimuth": bkw_azimuth,
        "line": get_geojson_from_geometry(line),
    }


@app.post("/routesearch/", tags=["geometry"], response_model=dict[str, Any])
async def calc_distance(
    start: FeatureCollection,
    goal: FeatureCollection,
    bbox: Annotated[list[float], Body()],
):
    start_geom = get_geometris_from_feature_collection(start)[0]
    goal_geom = get_geometris_from_feature_collection(goal)[0]

    graph = ox.graph_from_bbox(
        bbox=bbox, simplify=False, retain_all=True, network_type="drive"
    )
    start_node = ox.nearest_nodes(graph, get_x(start_geom), get_y(start_geom))
    end_node = ox.nearest_nodes(graph, get_x(goal_geom), get_y(goal_geom))

    shortest_path = ox.shortest_path(graph, start_node, end_node)
    coords = []
    for node_id in shortest_path:
        point = graph.nodes[node_id]
        coords.append([point["x"], point["y"]])

    return {"path": get_geojson_from_geometry(LineString(coords))}


"""
参考:
https://shapely.readthedocs.io/en/stable/reference/shapely.is_valid.html#shapely.is_valid
TODO: targetはFeatureCollectionで受けられるようにしたい。
"""


@app.post("/crosscheck/", tags=["geometry"], response_model=dict[str, bool])
async def execute_crosscheck(target: FeatureCollection):
    target_geom = get_geometris_from_feature_collection(target)[0]
    result = not is_valid(target_geom)
    return {"result": result}


# GeoJSONリクエスト用のモデル
class GeoJSONFeature(BaseModel):
    type: str
    properties: dict[str, Any]
    geometry: dict[str, Any]


class GeoJSONFeatureCollection(BaseModel):
    type: str
    features: list[GeoJSONFeature]


class GeoJSONRequest(BaseModel):
    geojson: GeoJSONFeatureCollection
    k: int  # 担当者数（リクエストに含める）

    def get_features(self) -> list[GeoJSONFeature]:
        return self.geojson.features


def _validate_points(request: GeoJSONRequest, n_points: int):
    # バリデーション: 拠点数がkより少ない場合はエラー
    if n_points < request.k:
        raise HTTPException(
            status_code=400,
            detail=f"拠点数({n_points})は担当者数({request.k})以上である必要があります。",
        )
    if request.k <= 0:
        raise HTTPException(
            status_code=400, detail="担当者数は1以上で指定してください。"
        )


def _extract_clustering_data(request: GeoJSONRequest):
    points_spatial = []
    altitudes = []
    for feature in request.get_features():
        if feature.geometry["type"] == "Point":
            lng, lat = feature.geometry["coordinates"][:2]
            alt = feature.properties.get("altitude", 0)
            if alt == 0 and len(feature.geometry["coordinates"]) > 2:
                alt = feature.geometry["coordinates"][2]
            points_spatial.append([lat, lng])
            altitudes.append([alt])

    return np.array(points_spatial), np.array(altitudes)    

@app.post(
    "/cluster/",
    tags=["geometry"],
    response_model=dict[str, Any],
    description="高度を考慮したクラスタリングを行い、傾斜負荷に応じた統計情報を含む凸包を返す。",
)
async def execute_kmeans_clustering(request: GeoJSONRequest):
    # データ抽出
    all_points_array, altitudes_array = _extract_clustering_data(request)
    points_size = len(all_points_array)

    _validate_points(request, points_size)

    # 標準化
    combined_data = np.hstack([all_points_array, altitudes_array]).astype(np.float64)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 高度ウェイトの強化
    # 高度方向の差を水平距離の 2.5倍 重く評価する
    # これにより「坂をまたぐ」クラスタリングを抑制する
    scaled_data_weighted = scaled_data.copy()
    scaled_data_weighted[:, 2] *= 2.5 

    # 制約付きk-meansの実行
    # 傾斜の激しいエリアから拠点が溢れることを許容するため、サイズに遊びを持たせる
    avg_size = points_size / request.k
    min_size = max(1, int(avg_size - 1))
    max_size = int(np.ceil(avg_size + 1))

    clf = KMeansConstrained(
        n_clusters=request.k,
        size_min=min_size,
        size_max=max_size,
        random_state=42
    )
    labels = clf.fit_predict(scaled_data_weighted)

    # GeoJSON構造の作成
    features = []
    for i in range(request.k):
        cluster_mask = (labels == i)
        cluster_data = all_points_array[cluster_mask]
        cluster_alts = altitudes_array[cluster_mask]
        
        # 傾斜負荷の計算 (高度の標準偏差 = そのエリアの起伏の激しさ)
        elevation_gain = np.max(cluster_alts) - np.min(cluster_alts) if len(cluster_alts) > 0 else 0
        slope_risk = np.std(cluster_alts) if len(cluster_alts) > 0 else 0

        # --- 図形生成ロジック (凸包 / 矩形 / 1点) ---
        if len(cluster_data) >= 3:
            try:
                hull = ConvexHull(cluster_data)
                hull_points = cluster_data[hull.vertices]
                polygon_coords = [
                    [[p[1], p[0]] for p in hull_points]
                    + [[hull_points[0][1], hull_points[0][0]]]
                ]
            except: # 同一線上の場合などは矩形にフォールバック
                min_lat, min_lng = cluster_data.min(axis=0)
                max_lat, max_lng = cluster_data.max(axis=0)
                polygon_coords = [[[min_lng, min_lat], [max_lng, min_lat], [max_lng, max_lat], [min_lng, max_lat], [min_lng, min_lat]]]

        elif len(cluster_data) == 2:
            min_lat, min_lng = cluster_data.min(axis=0)
            max_lat, max_lng = cluster_data.max(axis=0)
            polygon_coords = [[[min_lng, min_lat], [max_lng, min_lat], [max_lng, max_lat], [min_lng, max_lat], [min_lng, min_lat]]]

        else:
            lat, lng = cluster_data[0]
            offset = 0.0001 
            polygon_coords = [[[lng - offset, lat - offset], [lng + offset, lat - offset], [lng + offset, lat + offset], [lng - offset, lat + offset], [lng - offset, lat - offset]]]

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "worker_id": i + 1,
                    "point_count": int(len(cluster_data)),
                    "avg_altitude": round(float(np.mean(cluster_alts)), 2),
                    "elevation_gain": round(float(elevation_gain), 2),
                    "slope_risk": round(float(slope_risk), 2), # これが高いと傾斜が厳しい
                },
                "geometry": {"type": "Polygon", "coordinates": polygon_coords},
            }
        )

    return {"type": "FeatureCollection", "features": features}

