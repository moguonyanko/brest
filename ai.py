from functools import lru_cache
from fastapi import FastAPI, HTTPException, status, Depends
from typing_extensions import Annotated
from openai import AzureOpenAI, APIConnectionError
from config import AzureOpenAiSettings

app = FastAPI(
    title="Brest AI API",
    description="OpenAIの機能をREST APIで提供する。本APIではAzureOpenAIを使用する。",
    summary="Brest AI API by REST",
    version="0.0.1"
)

@lru_cache
def get_settings():
  return AzureOpenAiSettings()

def get_openai_client(az_settings: AzureOpenAiSettings) -> AzureOpenAI:
  client = AzureOpenAI(
      api_key=az_settings.key,
      api_version=az_settings.version,
      azure_endpoint=az_settings.endpoint
  )
  return client

@app.get("/simplechat/", tags=["ai"], response_model=dict[str, str])
async def simple_chat(text: str, settings: Annotated[AzureOpenAiSettings, 
                                                     Depends(get_settings)]):
  client = get_openai_client(settings)
  model = settings.model_name
  max_tokens = settings.max_tokens

  try:   
    response = client.completions.create(model=model, prompt=text, max_tokens=max_tokens)
  except APIConnectionError:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                        detail="Failed API connection")
  return {
    'result': response.choices[0].text
  }
