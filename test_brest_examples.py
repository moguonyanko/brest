from fastapi.testclient import TestClient

from main import brest_service, APP_ROOT

test_client = TestClient(brest_service)

def test_get_greeting():
    response = test_client.get(APP_ROOT + "greeting/ja")
    assert response.status_code == 200
    assert response.json() == {"greeting": "こんにちは"}

def test_register_greeting():
    response = test_client.post(APP_ROOT + "greeting/",
                                headers={
                                    "X-token": "langtoken"
                                },
                                json={
                                    "lang": "original",
                                    "greeting": "ハポー"
                                })
    assert response.status_code == 200
    #response_modelでGreetingオブジェクトを指定していも戻り値はJSONになっている。
    assert response.json() == {"lang": "original", "greeting": "ハポー"}
