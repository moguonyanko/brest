from fastapi import FastAPI, HTTPException, status
import os
from openai import AzureOpenAI, APIConnectionError

app = FastAPI(
    title="Brest AI API",
    description="OpenAIの機能をREST APIで提供する。本APIではAzureOpenAIを使用する。",
    summary="Brest AI API by REST",
    version="0.0.1"
)

def get_openai_client() -> AzureOpenAI:
  client = AzureOpenAI(
      api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
      api_version="2024-02-01",
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
      )
  return client

def get_open_ai_model_name() -> str:
  deployment_name='REPLACE_WITH_YOUR_DEPLOYMENT_NAME'
  return deployment_name
    
@app.get("/simplechat/", tags=["ai"], response_model=dict[str, str])
async def simple_chat(text: str):
  client = get_openai_client()
  model = get_open_ai_model_name()
  
  try:   
    response = client.completions.create(model=model, prompt=text, max_tokens=10)
  except APIConnectionError:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                        detail="Failed API connection")
  return {
    'result': response.choices[0].text
  }
