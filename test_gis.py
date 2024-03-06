from fastapi.testclient import TestClient
from gis import app

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
