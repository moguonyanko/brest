from pathlib import Path
from fastapi.testclient import TestClient
from genaiapi import app, get_genai_client, get_generate_image_model_name
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

test_client = TestClient(app)

'''
テキストのプロンプトからテキスト生成して返すテスト
'''
def test_generate_text():
  response = test_client.post("/generate/text/",
                              json={"contents":"What is one plus one?"})
  assert response.status_code == 200
  text = response.json()["results"]
  print(text)
  assert "two" in text.lower()

'''
ローカルの動画ファイルをインラインでアップロードして文字起こしできるかを確認するテスト
参考:
https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#inline-video
'''
def test_generate_text_from_inline_movie():
  with open(f"{Path.home()}/share/movie/samplenote.m4v", 'rb') as f:
    video_bytes = f.read()

  response = get_genai_client().models.generate_content(
      model='models/gemini-2.0-flash',
      contents=types.Content(
          parts=[
              types.Part(text='Can you summarize this video?'),
              types.Part(
                  inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
              )
          ]
      )
  )   

  assert response.text is not None

'''
ローカルの音声ファイルをインラインでアップロードして文字起こしできるかどうかのテスト
参考:
https://ai.google.dev/gemini-api/docs/audio?hl=ja&lang=python#inline-data
'''
def test_generate_text_from_inline_audio():
  with open(f"{Path.home()}/share/audio/sampleaudio.m4a", 'rb') as f:
      audio_bytes = f.read()

  response = get_genai_client().models.generate_content(
    model='gemini-2.0-flash',
    contents=[
      'Describe this audio clip',
      types.Part.from_bytes(
        data=audio_bytes,
        mime_type='audio/mp4',
      )
    ]
  )

  assert response.text is not None

'''
ローカルのPDFファイルから要約を生成するテスト
https://ai.google.dev/gemini-api/docs/document-processing?hl=ja&lang=python#local-pdfs
'''
def test_generate_text_from_local_pdf():
  client = get_genai_client()

  # Retrieve and encode the PDF byte
  filepath = Path(f"{Path.home()}/share/doc/sampledocument.pdf")

  prompt = "Summarize this document"
  response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents=[
        types.Part.from_bytes(
          data=filepath.read_bytes(),
          mime_type='application/pdf',
        ),
        prompt])
  
  assert response.text is not None
