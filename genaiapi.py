import json
from io import BytesIO
from typing import Union, Annotated, Any
from google import genai
from PIL import Image
from fastapi import FastAPI, HTTPException, status, Body, Depends
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
    config = json.load(f)

client = genai.Client(api_key=config['api_key'])
model_name = config['model_name']

class GenerationResultText(BaseModel):
  text: str

#TODO: types.GenerateContentConfigとどう併用するのか？
common_config = {
                'response_mime_type': 'application/json',
                'response_schema': GenerationResultText
            }

@app.post("/generate/text/", tags=["ai"], response_model=GenerationResultText)
async def generate_text(body: dict):
    response = client.models.generate_content(
        model=model_name,
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
        response = client.models.generate_content(
            model="gemini-2.0-flash",
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
    chat = client.chats.create(model=model_name)
    while True:
        user_message = await websocket.receive_text()
        response = chat.send_message(user_message)
        await websocket.send_text(response.text)
        