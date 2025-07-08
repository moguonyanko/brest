from pathlib import Path
from fastapi.testclient import TestClient
from genaiapi import app, get_genai_client
from google.genai import types
from PIL import Image
from io import BytesIO
import wave
import pytest
import soundfile as sf
import librosa
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, UrlContext
import asyncio

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

def write_wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
  """
  生成された音声データをローカルで確認できるようにするために書き出す関数です。
  """
  with wave.open(filename, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(pcm)

SAMPLE_SPEECH_MESASGE = "みなさんおはようございます。ほんじつもがんばりましょう。"

def test_generate_speech():
    """
    音声生成APIのテストです。

    参考:
    https://ai.google.dev/gemini-api/docs/speech-generation?hl=ja
    """
    response = get_genai_client().models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents=SAMPLE_SPEECH_MESASGE,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                # voice_nameは規定されている値を指定しないとHTTPエラーとなる。
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Puck')
            )
        )
    ))

    data = response.candidates[0].content.parts[0].inline_data.data    

    assert data is not None
    assert len(data) > 0

    filepath = f"{Path.home()}/share/audio/samplespeech.wav"

    write_wave_file(filepath, data)

def wavefile_to_pcmbytes(filepath: str, load_samplerate: int):
  buffer = BytesIO()
  data, samplerate = librosa.load(filepath, sr=load_samplerate)
  sf.write(buffer, data, samplerate, format='RAW', subtype='PCM_16')
  buffer.seek(0)
  audio_bytes = buffer.read()
  return audio_bytes

@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_generate_text_from_speech_file_by_live_api():
  """
  LiveAPIを使って音声ファイルから文字を取得します。

  _session.receive()から結果が返されない問題によりテスト成功しません。_

  **参考**

  https://ai.google.dev/gemini-api/docs/live-guide?hl=ja#send-receive-audio
  """
  client = get_genai_client()
  model = "gemini-live-2.5-flash-preview"
  config = {"response_modalities": ["AUDIO"]}
  filepath = f"{Path.home()}/share/audio/samplespeech.wav"

  async with client.aio.live.connect(model=model, config=config) as session:
    load_samplerate = 16000
    audio_bytes = wavefile_to_pcmbytes(filepath=filepath, load_samplerate=load_samplerate)

    await session.send_realtime_input(
       audio=types.Blob(data=audio_bytes, mime_type=f"audio/pcm;rate={load_samplerate}")
    )

    response_text = []
    async for response in session.receive():
      if response.text is not None:
        print(response.text) # ここに入ってこないしタイムアウトもしない。
        assert response.text is not None
        assert len(response.text) > 0
        response_text.append(response.text)

    joined_response_text = "".join(response_text)      
    print(joined_response_text)
    # Live APIは応答を返してくるのでSAMPLE_SPEECH_MESASGEと比較しても意味がない。
    assert joined_response_text is not None 

def test_extract_url_context():
  """
  指定されたURLの内容を読み取って処理を行うAPIのテストです。

  **参考**

  https://ai.google.dev/gemini-api/docs/url-context?hl=ja
  """
  client = get_genai_client()
  model_id = "gemini-2.5-flash"
  target_url = "https://ipqcache2.shufoo.net/c/2025/06/17/c/5852566107125/index/img/chirashi.pdf?shopId=2206&chirashiId=5852566107125"

  tools = []
  tools.append(Tool(url_context=UrlContext))
  tools.append(Tool(google_search=GoogleSearch))

  response = client.models.generate_content(
      model=model_id,
      contents=f"{target_url}に記載されている商品の割引情報を抽出してください",
      config=GenerateContentConfig(
          tools=tools,
          response_modalities=["TEXT"],
      )
  )

  for each in response.candidates[0].content.parts:
      print(each.text)
  # get URLs retrieved for context
  print(response.candidates[0].url_context_metadata)

def assert_valid_wave_file(filepath, channels, sampwidth, framerate):
  with wave.open(filepath, "rb") as check_wf:
    n_frames = check_wf.getnframes()
    print(f"WAV情報: チャンネル数={check_wf.getnchannels()}, サンプル幅={check_wf.getsampwidth()}, フレームレート={check_wf.getframerate()}, 総フレーム数={n_frames}")
          
    # 設定した値と一致するか確認することも可能
    assert check_wf.getnchannels() == channels
    assert check_wf.getsampwidth() == sampwidth
    assert check_wf.getframerate() == framerate

    assert n_frames > 0, "WAVファイルに音声フレームが含まれていません。"
    print(f"音声の長さ: {n_frames / check_wf.getframerate():.2f} 秒")

@pytest.mark.asyncio
async def test_write_audio_file_from_text_with_live_api():
  """
  LiveAPIを使ってテキストをwaveファイルに書き出します。

  __書き出したwavファイルに音声フレームが含まれておらずアサーションエラーになります。__

  **参考**

  https://ai.google.dev/gemini-api/docs/live-guide?hl=ja#send-receive-audio
  """
  client = get_genai_client()
  model = "gemini-2.5-flash-preview-native-audio-dialog"
  config = {"response_modalities": ["AUDIO"]}
  filepath = f"{Path.home()}/share/audio/samplespeech_from_text.wav"

  async with client.aio.live.connect(model=model, config=config) as session:
    channels = 1
    sampwidth = 2
    framerate = 24000
    wf = wave.open(filepath, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)

    message = "これからもよろしくお願いいたします"
    await session.send_client_content(
        turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
    )

    async for response in session.receive():
      if response.data is not None:
        wf.writeframes(response.data)
    
    wf.close()
    
    assert_valid_wave_file(filepath=filepath, channels=channels, 
                           sampwidth=sampwidth, framerate=framerate)

@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_generate_audio_from_user_audio_with_live_api():
  """
  LiveAPIを使ってユーザーの音声ファイルから音声を生成します。

  session.receive()から応答が返されずタイムアウトになります。
  """
  client = get_genai_client()
  model = "gemini-2.5-flash-preview-native-audio-dialog"
  config = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "You are a helpful assistant and answer in a friendly tone.",
  }

  async with client.aio.live.connect(model=model, config=config) as session:
    framerate = 24000
    input_filepath = f"{Path.home()}/share/audio/samplespeech.wav"
    output_filepath = f"{Path.home()}/share/audio/samplespeech_output.wav"

    audio_bytes = wavefile_to_pcmbytes(filepath=input_filepath, load_samplerate=framerate)

    await session.send_realtime_input(
      audio=types.Blob(data=audio_bytes, mime_type=f"audio/pcm;rate={framerate}")
    )

    channels = 1
    sampwidth = 2
    wf = wave.open(output_filepath, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)

    async for response in session.receive():
      if response.data is not None:
        wf.writeframes(response.data)
        if response.server_content.model_turn is not None:
          print(response.server_content.model_turn.parts[0].inline_data.mime_type)        

    wf.close()  
    assert_valid_wave_file(filepath=output_filepath, channels=channels, 
                           sampwidth=sampwidth, framerate=framerate)
