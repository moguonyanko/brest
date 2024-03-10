from typing import Union, Tuple
from fastapi import FastAPI, HTTPException, status

app = FastAPI(
    title="Brest GIS API",
    description="GISの計算機能をREST APIで提供する。",
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

@app.post("/linecrosscheck/", tags=["geometry"])
async def check_cross_lines(line1: dict, line2: dict):
    return {"result": 0}
