from fastapi.testclient import TestClient
from genaiapi import app

test_client = TestClient(app)

def test_generate_text():
  response = test_client.post("/generate/text/",
                              json={"contents":"What is one plus one?"})
  assert response.status_code == 200
  text = response.json()["results"]
  print(text)
  assert "two" in text.lower()
