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

def test_get_sample_xml():
    response = test_client.get(APP_ROOT + "samplexml/sample")
    
    with open("sample/sample.xml", "r") as xml_file:
        expected_content = xml_file.read()

    assert response.status_code == 200
    assert response.text == expected_content

def test_inject_sql():
    response = test_client.post(APP_ROOT + "inject_sql/",
                                json={
                                    "sql": "SELECT 1"
                                })
    assert response.status_code == 200
    assert response.json() == {"result": "1"}
