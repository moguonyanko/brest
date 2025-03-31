import json
import requests
from io import BytesIO
from typing import Union, Annotated, Any
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

# Gemini APIの初期化
with open('genaiapi_config.json', 'r') as f:
    genaiapi_config = json.load(f)

def get_genai_client():
    return genai.Client(api_key=genaiapi_config['api_key'])

def get_generate_text_model_name() -> str:
    return genaiapi_config['model_name']['generate_text']

def get_generate_image_model_name() -> str:
    return genaiapi_config['model_name']['generate_image']

def get_generate_vision_model_name() -> str:
    return genaiapi_config['model_name']['vision']

class GenerationResultText(BaseModel):
  text: str

#TODO: types.GenerateContentConfigとどう併用するのか？
common_config = {
                'response_mime_type': 'application/json',
                'response_schema': GenerationResultText
            }

@app.post("/generate/text/", tags=["ai"], response_model=GenerationResultText)
async def generate_text(body: dict):
    response = get_genai_client().models.generate_content(
        model=get_generate_text_model_name(),
        contents=body['contents'],
        config=common_config
    )
    return response.parsed

@app.post("/generate/text-from-image/", tags=["ai"], response_model=GenerationResultText)
async def generate_test_from_image(
    file: Annotated[UploadFile, File(description="プロンプトに渡す画像です。")]
):
    try:
        request_file_content = await file.read()
        image = Image.open(BytesIO(request_file_content))
        response = get_genai_client().models.generate_content(
            model=get_generate_text_model_name(),
            contents=[image, "画像について説明してください。"],
            config=common_config
        )       
        # 回答は英語で返されてしまう。
        return response.parsed 
    except Exception:
        raise HTTPException(status_code=500, detail='画像による問い合わせに失敗しました。')
    finally:
        file.file.close()
        image.close()

#TODO: closeされるとエラーが発生してしまう。
@app.websocket("/generate/talk/")
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
        
@app.post("/generate/image/", tags=["ai"], 
    responses = {
        200: {
            "content": {"image/png": {}}
        },
        400: {
            "content": {"text/plain": {}}
        }
    },
    response_class=Response)
async def generate_image(body: dict):   
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
            return Response(content=image_bytes.getvalue(), media_type='image/png')

@app.post("/generate/text-from-image-url/", tags=["ai"], response_model=GenerationResultText)
async def generate_test_from_image_url(body: dict):
    try:
        image = requests.get(body['url'])
        content_type = image.headers.get('Content-Type')
        response = get_genai_client().models.generate_content(
            model=get_generate_vision_model_name(),
            contents=["画像について説明してください。", 
                      types.Part.from_bytes(data=image.content, mime_type=content_type)],
            config=common_config
        )       
        # 回答は英語になってしまう。
        return response.parsed 
    except Exception:
        raise HTTPException(status_code=500, detail='画像による問い合わせに失敗しました。')

'''
参考:
https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#bbox
'''
@app.post("/generate/bouding-box-from-image/", tags=["ai"], response_model=str)
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

