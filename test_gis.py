from fastapi.testclient import TestClient
from gis import app

#Webアプリケーションが起動していなくてもTestClientによるテストは実行できる。
test_client = TestClient(app)

def test_get_hellogis():
    response = test_client.get("/hellogis/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Brest GIS!"}

def test_get_point_side_of_line():
    response = test_client.post("/pointsideofline/",
                                json={"point":{"type":"Feature","properties":{},"geometry":{"coordinates":[139.751363304702,35.65771000179585],"type":"Point"}},"line":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.7551995346509,35.66006748781244],[139.75693265376594,35.65465624195437]],"type":"LineString"}}]}})
    assert response.status_code == 200
    assert response.json() == {"side": -1}

def test_check_cross_lines():
    response = test_client.post("linecrosscheck",
                                json={"line1":{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75644916016995,35.6580874134024],[139.75590014523328,35.655063923545256]],"type":"LineString"}}]},"line2":{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75800034522894,35.657506798937305],[139.7541398116271,35.6560694059431]],"type":"LineString"}}})
    assert response.status_code == 200
    assert response.json() == {"result": True}

def test_calc_convexhull():
    response = test_client.post("convexhull", 
                                json={"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[139.75463668298983,35.65737926565541],[139.75712903651032,35.65761292845477],[139.75978130764446,35.65497571395987],[139.75199640429526,35.655462293990055],[139.75199640429526,35.655462293990055],[139.75953804242914,35.65869735639879],[139.75760608843967,35.66130862409193]],"type":"MultiPoint"}}]})
    assert response.status_code == 200
    assert response.json() == {"result":{"type":"Polygon","coordinates":[[[139.75978130764446,35.65497571395987],[139.75199640429526,35.655462293990055],[139.75760608843967,35.66130862409193],[139.75953804242914,35.65869735639879],[139.75978130764446,35.65497571395987]]]}}
    