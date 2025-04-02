from fastapi.testclient import TestClient
from genaiapi import app, get_genai_client, get_generate_image_model_name
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

test_client = TestClient(app)

def test_generate_text():
  response = test_client.post("/generate/text/",
                              json={"contents":"What is one plus one?"})
  assert response.status_code == 200
  text = response.json()["results"]
  print(text)
  assert "two" in text.lower()
