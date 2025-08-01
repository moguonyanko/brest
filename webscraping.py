"""WebスクレイピングのAPIを提供します。

参考資料:
PythonによるWebスクレイピング 第3版
https://www.oreilly.co.jp/books/9784814401222/
"""

import json
import requests
import time
from io import BytesIO
from typing import Annotated
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from fastapi import FastAPI, HTTPException, Body, Response, status, Depends
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
from urllib.error import URLError
from bs4 import BeautifulSoup

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


def read_all_contents(url: str) -> str:
    try:
        contents = urlopen(url)
    except URLError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.reason)
    return contents.read()


@app.get("/pagecontents/", tags=["url"], response_model=dict[str, str])
async def ws_get_page_contents(url: str):
    return {"contents": read_all_contents(url)}


def get_title(contents) -> str:
    try:
        bs = BeautifulSoup(contents, "html.parser")
        return bs.body.h1.contents[0]
    except AttributeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.args[0]
        )


@app.get("/pagetitle/", tags=["url"], response_model=dict[str, str])
async def ws_get_page_title(url: str):
    contents = read_all_contents(url)
    return {"title": get_title(contents)}
