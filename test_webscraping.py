from fastapi.testclient import TestClient
from webscraping import app
import pytest

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


def test_error_http_exception_when_not_found_title():
    """存在しないページのタイトルを得ようとした時に500エラーになることのテストです。"""
    response = test_client.get("/pagetitle/?url=http://localhost/errorsite")
    assert response.status_code == 400


def test_get_page_title():
    response = test_client.get("/pagetitle/?url=http://localhost/webxam")
    assert response.status_code == 200
    title = response.json()["title"]
    assert title is not None
    assert title == "WebXam"
