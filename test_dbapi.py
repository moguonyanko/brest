from fastapi.testclient import TestClient
from dbapi import app

test_client = TestClient(app)

def test_inject_sql():
  response = test_client.post("/injectsql/",
                              json={"sql":"SELECT count(*) FROM fruits WHERE id = 0"})
  assert response.status_code == 200
  assert response.json() == {"results": [[1]]}

if __name__ == '__main__':
  test_inject_sql()  
