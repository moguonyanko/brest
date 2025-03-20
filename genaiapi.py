import json
from typing import Union, Annotated, Any
from google import genai
from fastapi import FastAPI, HTTPException, status, Body, Depends
from pydantic import BaseModel

app = FastAPI(
    title="Brest Generative AI API",
    description="生成AIを操作する機能をAPIで提供する。",
    summary="Gen AI API by REST",
    version="0.0.1"
)

#Gemini APIの初期化
with open('genaiapi_config.json', 'r') as f:
  config = json.load(f)

client = genai.Client(api_key=config['api_key'])
model_name = config['model_name']

@app.post("/generate/text/", tags=["ai"], response_model=dict[str, Any])
def generate_text(body: dict):
  response = client.models.generate_content(
      model=model_name, 
      contents=body['contents']
  )
  return {
    "results": response.text
  }
