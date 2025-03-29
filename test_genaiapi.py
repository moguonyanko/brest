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

'''
参考:
https://ai.google.dev/gemini-api/docs/image-generation?hl=ja
'''
def test_generate_image():
  client = get_genai_client()

  contents = ('Create photorealistic, fresh images of apples.')

  response = client.models.generate_content(
      model=get_generate_image_model_name(),
      contents=contents,
      config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
      )
  )

  for part in response.candidates[0].content.parts:
    if part.text is not None:
      print(part.text)
    elif part.inline_data is not None:
      image = Image.open(BytesIO((part.inline_data.data)))
      image.save('gemini-native-image.png')
      image.show()
