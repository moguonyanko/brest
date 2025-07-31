import json
import requests
import time
from io import BytesIO
from typing import Annotated
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, UrlContext
from PIL import Image
from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask
from fastapi import File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from urllib.parse import quote
import wave
from pathlib import Path
import uuid
import os
from urllib.request import urlopen

app = FastAPI(
    title="Brest Web Scraping API",
    description="WebスクレイピングのAPIで提供する。",
    summary="Web Scraping API by REST",
    version="0.0.1",
)

app_base_path = "/webscraping"


@app.get("/hellowebscraping/", tags=["test"])
async def get_hello_webscraping():
    return {"message": "Hello Brest Web Scraping!"}


@app.get("/pagecontents/", tags=["url"])
async def get_page_contents(url: str):
    contents = urlopen(url)
    return {"contents": contents.read()}
