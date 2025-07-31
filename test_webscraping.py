from fastapi.testclient import TestClient
from webscraping import app

test_client = TestClient(app)


def test_get_hellogis():
    response = test_client.get("/hellowebscraping/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Brest Web Scraping!"}


def test_get_page_contents():
    response = test_client.get("/pagecontents/?url=http://localhost/")
    assert response.status_code == 200
    contents = response.json()["contents"]
    assert contents is not None
