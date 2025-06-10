from pathlib import Path
from fastapi.testclient import TestClient
from genaiapi import app, get_genai_client, get_generate_image_model_name
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64

test_client = TestClient(app)

def test_generate_text():
  '''
  テキストのプロンプトからテキスト生成して返すテスト
  '''
  response = test_client.post("/generate/text/",
                              json={"contents":"What is one plus one?"})
  assert response.status_code == 200
  text = response.json()["text"]
  print(text)
  assert "two" in text.lower()

def test_generate_text_from_inline_movie():
  '''
  ローカルの動画ファイルをインラインでアップロードして文字起こしできるかを確認するテスト
  参考:
  https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#inline-video
  '''
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

def test_generate_text_from_inline_audio():
  '''
  ローカルの音声ファイルをインラインでアップロードして文字起こしできるかどうかのテスト
  参考:
  https://ai.google.dev/gemini-api/docs/audio?hl=ja&lang=python#inline-data
  '''
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

def test_generate_text_from_local_pdf():
  '''
  ローカルのPDFファイルから要約を生成するテスト
  https://ai.google.dev/gemini-api/docs/document-processing?hl=ja&lang=python#local-pdfs
  '''
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

def test_generate_image_from_text():
  '''
  テキストから画像生成できるかどうかを確認するための関数です。
  '''
  client = get_genai_client()
  contents = ('Cats that turn into humans and rule the earth')

  response = client.models.generate_content(
      model="gemini-2.0-flash-exp-image-generation",
      contents=contents,
      config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
      )
  )

  for part in response.candidates[0].content.parts:
    if part.text is not None:
      print(part.text)
    elif part.inline_data is not None:
      print(part.inline_data.mime_type)
      bytes = BytesIO(part.inline_data.data)
      image = Image.open(bytes)
      image.save(f"{Path.home()}/share/img/genai/gemini-native-image.png")
      # image.show()  

  assert image is not None  

def test_generate_thinking_result():
  """
  思考するAPIでテキストを生成できるかのテストです。

  参考:
  https://ai.google.dev/gemini-api/docs/thinking?hl=ja
  """
  client = get_genai_client()

  response = client.models.generate_content(
      model="gemini-2.5-flash-preview-04-17",
      contents="Explain the Occam's Razor concept and provide everyday examples of it",
      config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig()
      )
  )

  assert response.text is not None

def test_generate_speech():
    """
    音声生成APIのテストです。

    参考:
    https://ai.google.dev/gemini-api/docs/speech-generation?hl=ja
    """
    response = get_genai_client().models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents="みなさんおはようございます。本日もよろしくお願いいたします。",
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                # voice_nameは規定されている値を指定しないとHTTPエラーとなる。
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Kore')
            )
        )
    ))

    data = response.candidates[0].content.parts[0].inline_data.data    

    assert data is not None
