import json
import requests
import time
from io import BytesIO
from typing import Union, Annotated, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from PIL import Image
from fastapi import FastAPI, HTTPException, status, Body, Depends, Response
from fastapi.responses import StreamingResponse
from fastapi import File, UploadFile, WebSocket
from pydantic import BaseModel

app = FastAPI(
    title="Brest Generative AI API",
    description="生成AIを操作する機能をAPIで提供する。",
    summary="Gen AI API by REST",
    version="0.0.1"
)

app_base_path = '/generate'

def load_config_file(path: str):
    with open(path, 'r') as f:
        return json.load(f)

# Gemini APIの共通設定読み込み
genaiapi_config = load_config_file(path='genaiapi_config.json')

# Gemini APIのモデル設定読み込み
genaiapi_models = load_config_file(path='genaiapi_models.json')
genaiapi_model_names = genaiapi_models['model_name']

# Gemini APIのキー読み込み
api_keys = load_config_file(path='genaiapi_keys.json')['api_keys']
# APIキーを使い分ける必要が生じたらcommon以外を参照できるように以下のコードを修正する。
api_key = api_keys['common']

def get_genai_client():
    return genai.Client(api_key=api_key)

'''
設定ファイルからモデル名を取得する関数群
'''
def get_generate_text_model_name() -> str:
    return genaiapi_model_names['generate_text']

def get_generate_image_model_name() -> str:
    return genaiapi_model_names['generate_image']

def get_generate_vision_model_name() -> str:
    return genaiapi_model_names['vision']

def get_generate_transcription_movie_model_name() -> str:
    return genaiapi_model_names['transcription_movie']

def get_generate_transcription_movie_inline_model_name() -> str:
    return genaiapi_model_names['transcription_movie_inline']

def get_generate_transcription_audio_inline_model_name() -> str:
    return genaiapi_model_names['transcription_audio_inline']

def get_model_name_summarize_document() -> str:
    return genaiapi_model_names['summarize_document']

def get_model_name_text_embedding() -> str:
    return genaiapi_model_names['text_embedding']

'''
生成結果をJSONの形式で返すためのクラス
'''
class GenerationResultText(BaseModel):
  text: str

@app.post(f"{app_base_path}/text/", tags=["ai"], response_model=GenerationResultText)
async def generate_text(body: dict):
    response = get_genai_client().models.generate_content(
        model=get_generate_text_model_name(),
        contents=body['contents'],
        config={
                'response_mime_type': 'application/json',
                'response_schema': GenerationResultText
            }
    )
    return response.parsed

@app.post(f"{app_base_path}/text-from-image/", tags=["ai"], response_model=GenerationResultText)
async def generate_test_from_image(
    file: Annotated[UploadFile, File(description="プロンプトに渡す画像です。")]
):
    try:
        request_file_content = await file.read()
        image = Image.open(BytesIO(request_file_content))
        response = get_genai_client().models.generate_content(
            model=get_generate_text_model_name(),
            contents=[image, "画像について説明してください。"],
            config={
                'response_mime_type': 'application/json',
                'response_schema': GenerationResultText
            }
        )       
        # 回答は英語で返されてしまう。
        return response.parsed 
    except Exception:
        raise HTTPException(status_code=500, detail='画像による問い合わせに失敗しました。')
    finally:
        file.file.close()
        image.close()

#TODO: closeされるとエラーが発生してしまう。
@app.websocket(f"{app_base_path}/talk/")
async def talk_generative_ai(websocket: WebSocket):
    await websocket.accept()
    chat = get_genai_client().chats.create(
        model=get_generate_text_model_name(),
        config=types.GenerateContentConfig(
            max_output_tokens=genaiapi_config['max_output_tokens'],
            temperature=genaiapi_config['temperature']
    ))
    while True:
        user_message = await websocket.receive_text()
        response = chat.send_message(user_message)
        await websocket.send_text(response.text)
        
@app.post(f"{app_base_path}/image/", tags=["ai"], 
    responses = {
        200: {
            "content": {"image/*": {}}
        },
        500: {
            "content": {"text/plain": {}}
        }
    },
    response_class=Response)
async def generate_image(body: Annotated[dict, Body(
                        examples=[
                            {
                                "contents": "Generate an image of an apple."
                            }
                        ]
                    )]):   
    """
    リクエストされたテキストから画像を生成します。
    TODO: 画像生成時か生成結果をBytesIOのコンストラクタに渡した後でエラーになってしまう。
    """
    try:
        response = get_genai_client().models.generate_content(
            model=get_generate_image_model_name(),
            contents=body['contents'],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )

        # 最初の結果だけ返している。
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                return Response(content=part.text, media_type='text/plain')            
            elif part.inline_data is not None:
                image_bytes = BytesIO((part.inline_data.data))
                image_bytes.seek(0)
                return Response(content=image_bytes.getvalue(), 
                                media_type=part.inline_data.mime_type)
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Generative Error: {err=}, {type(err)=}")

@app.post(f"{app_base_path}/text-from-image-url/", tags=["ai"], response_model=GenerationResultText)
async def generate_test_from_image_url(body: dict):
    try:
        image = requests.get(body['url'])
        content_type = image.headers.get('Content-Type')
        response = get_genai_client().models.generate_content(
            model=get_generate_vision_model_name(),
            contents=["画像について説明してください。", 
                      types.Part.from_bytes(data=image.content, mime_type=content_type)],
            config={
                'response_mime_type': 'application/json',
                'response_schema': GenerationResultText
            }
        )       
        # 回答は英語になってしまう。
        return response.parsed 
    except Exception:
        raise HTTPException(status_code=500, detail='画像による問い合わせに失敗しました。')

'''
参考:
https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#bbox
'''
@app.post(f"{app_base_path}/bouding-box-from-image/", tags=["ai"], response_model=str)
async def generate_bounding_box_from_image(
    file: Annotated[UploadFile, File(description="プロンプトに渡す画像です。")]
):
    try:
        request_file_content = await file.read()
        image = Image.open(BytesIO(request_file_content))
        prompt = (
        "Return a bounding box for each of the objects in this image "
        "in [ymin, xmin, ymax, xmax] format.")        
        response = get_genai_client().models.generate_content(
            model=get_generate_text_model_name(),
            contents=[image, prompt],
            config={
                'response_mime_type': 'application/json'
            }
        )      
        return response.text
    except Exception:
        raise HTTPException(status_code=500, detail='画像による問い合わせに失敗しました。')
    finally:
        file.file.close()
        image.close()

'''
参考:
https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#prompt-video
'''
@app.post(f"{app_base_path}/transcription-from-movie/", tags=["ai"], response_model=str)
async def generate_transcription_from_movie(
    file: Annotated[UploadFile, File(description="プロンプトに渡す動画です。")]
):
    try:
        client = get_genai_client()
        video_file = client.files.upload(file=file.file,
                                         config={
                                             'mime_type': file.content_type
                                         })

        # アップロードされた動画が利用可能になるまで待機する。
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        prompt_for_movie_summary = (
        "Please provide the Japanese text you would like me to translate into English."
        "Once you provide the Japanese text, I will also generate a summary of its content.")        

        response = client.models.generate_content(
            model=get_generate_transcription_movie_model_name(),
            contents=[video_file, prompt_for_movie_summary],
            config=types.GenerateContentConfig(**genaiapi_config)
        )      
        return response.text
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Prompt Error: {err=}, {type(err)=}")

'''
動画をインラインでアップロードして要約を生成します。
'''
@app.post(f"{app_base_path}/transcription-inline-from-movie/", tags=["ai"], response_model=str)
async def generate_transcription_inline_from_movie(
    file: Annotated[UploadFile, File(description="プロンプトに渡す動画です。")]
):
    try:
        client = get_genai_client()

        prompt_for_movie_summary = (
        "Please provide the Japanese text you would like me to translate into English."
        "Once you provide the Japanese text, I will also generate a summary of its content.")        

        video_bytes = await file.read()

        response = client.models.generate_content(
            model=get_generate_transcription_movie_inline_model_name(),
            contents=types.Content(
                parts=[
                    types.Part(text=prompt_for_movie_summary),
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, 
                                            mime_type=file.content_type)
                    )
                ]
            ),
            config=types.GenerateContentConfig(**genaiapi_config)
        )
        return response.text
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Generation Error: {err=}, {type(err)=}")
    
'''
音声ファイルをインラインでAPIに渡して文字起こしかつ要約します。
'''    
@app.post(f"{app_base_path}/transcription-inline-from-audio/", tags=["ai"], response_model=str)
async def generate_transcription_inline_from_auido(
    file: Annotated[UploadFile, File(description="プロンプトに渡す音声です。")]
):
    client = get_genai_client()

    prompt_for_audio_summary = "Please transcribe the audio file into Japanese and then summarize it."

    audio_bytes = await file.read()

    try:
        response = client.models.generate_content(
            model=get_generate_transcription_audio_inline_model_name(),
            contents=[
                prompt_for_audio_summary,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=file.content_type,
                )
            ],
            config=types.GenerateContentConfig(**genaiapi_config)
        )
        return response.text
    except Exception as err:
        raise HTTPException(status_code=500, 
            detail=f"Generation Error: {err=}, {type(err)=}")


@app.post(f"{app_base_path}/summarization-from-document/", tags=["ai"], response_model=str,
          description='アップロードされたドキュメントから要約を生成します。')
async def generate_summarization_from_document(
    file: Annotated[UploadFile, File(description="処理対象ドキュメントです。")]
):
    try:
        # ドキュメントが大きい場合は読み込み方法を再考する。
        doc_bytes = await file.read() 

        prompt = "文書の要約を生成してください"
        response = get_genai_client().models.generate_content(
            model=get_model_name_summarize_document(),
            contents=[
                types.Part.from_bytes(
                data=doc_bytes,
                mime_type=file.content_type,
                ), prompt],
                config=types.GenerateContentConfig(**genaiapi_config))
        
        return response.text
    except Exception as err:
        raise HTTPException(status_code=500, 
            detail=f"Generation Error: {err=}, {type(err)=}")

def calculate_pairwise_similarities(embeddings: list):
    """
    複数の埋め込みベクトル間のペアワイズなコサイン類似度を計算します。

    Args:
        embeddings (list of list or numpy.ndarray): 埋め込みベクトルのリスト。

    Returns:
        list of float: すべてのペアのコサイン類似度のリスト。
    """
    num_embeddings = len(embeddings)
    similarities = []
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            emb1 = np.array(embeddings[i].values).reshape(1, -1)
            emb2 = np.array(embeddings[j].values).reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(similarity)
    return similarities

def aggregate_similarity(pairwise_similarities, aggregation_method="mean"):
    """
    ペアワイズな類似度を特定の方法で集約します。

    Args:
        pairwise_similarities (list of float): ペアワイズな類似度のリスト。
        aggregation_method (str): 集約方法 ("mean", "min", "median"のいずれか)。
        デフォルトは "mean"。

    Returns:
        float: 集約された類似度。
    """
    if not pairwise_similarities:
        return 0.0

    if aggregation_method == "mean":
        return np.mean(pairwise_similarities)
    elif aggregation_method == "min":
        return np.min(pairwise_similarities)
    elif aggregation_method == "median":
        return np.median(pairwise_similarities)
    else:
        raise ValueError("'mean', 'min', 'median' のいずれかを指定してください。")

@app.post(f"{app_base_path}/text-similarity/", tags=["ai"], response_model=float,
          description='テキストの埋め込みを利用してテキストの類似性を取得します。')
async def generate_text_similarity(body: Annotated[dict, 
                                    Body(
                        examples=[
                            {
                                "contents": [
                                    "Hello, World",
                                    "こんにちは、世界"
                                ]
                            }
                        ]
                    )]):
    try:
        contents = body['contents']

        response = get_genai_client().models.embed_content(
            model=get_model_name_text_embedding(),
            contents=contents,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )

        pairwise_similarities = calculate_pairwise_similarities(response.embeddings)
        return aggregate_similarity(pairwise_similarities)
    except Exception as err:
        raise HTTPException(status_code=500, 
            detail=f"Generation Error: {err=}, {type(err)=}")
    