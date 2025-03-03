from fastapi.testclient import TestClient
from dbapi import app

test_client = TestClient(app)

def test_inject_sql():
  response = test_client.post("/injectsql/",
                              json={"sql":"SELECT count(*) FROM postal_codes"})
  assert response.status_code == 200
  assert response.json() == {"results": [[124434]]}
