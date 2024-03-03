from fastapi.testclient import TestClient
from gis import app

test_client = TestClient(app)

def test_get_hellogis():
    response = test_client.get("/hellogis/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Brest GIS!"}
