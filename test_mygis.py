from fastapi.testclient import TestClient
from mygis import app

test_client = TestClient(app)

def test_get_hellomygis():
    response = test_client.get("/hellomygis/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello My GIS!"}
