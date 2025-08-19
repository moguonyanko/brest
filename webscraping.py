"""WebスクレイピングのAPIを提供します。

参考資料:
PythonによるWebスクレイピング 第3版
https://www.oreilly.co.jp/books/9784814401222/
"""

import re
import json
import requests
import time
from io import BytesIO
from typing import Annotated
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from fastapi import FastAPI, HTTPException, Body, Response, status, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask
from fastapi import File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from urllib.parse import quote
import wave
from pathlib import Path
import uuid
import os
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError
from bs4 import BeautifulSoup
from pypdf import PdfReader
import tempfile
from nltk import word_tokenize, Text, pos_tag
from PIL import Image
import pytesseract

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


def parse_page_contents(contents):
    """
    urlopenで取得した結果をパースします。現在はBeautifulSoupとhtml.parserを常に使います。
    他の方法でパースしたい場合は設定ファイルなどで切り替えられるようにします。
    """
    result = BeautifulSoup(contents, "html.parser")
    return result


def get_title(contents) -> str:
    try:
        parsed_contents = parse_page_contents(contents)
        return parsed_contents.body.h1.get_text()
    except AttributeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.args[0]
        )


def get_image_src_list(contents, image_format) -> list[str]:
    parsed_contents = parse_page_contents(contents)
    if image_format is not None:
        images = parsed_contents.find_all(
            "img", {"src": re.compile(rf".*\.{image_format}$")}
        )
    else:
        images = parsed_contents.find_all("img")

    srclist = []
    for img in images:
        src = img.get("src")
        srclist.append(src)
    return srclist


@app.get("/pagetitle/", tags=["url"], response_model=dict[str, str])
async def ws_get_page_title(url: str):
    contents = read_all_contents(url)
    return {"title": get_title(contents)}


@app.get("/pageimgsrclist/", tags=["url"], response_model=dict[str, list[str]])
async def ws_get_page_image_src_list(url: str, format: str = None):
    contents = read_all_contents(url)
    imgsrclist = get_image_src_list(contents, format)
    return {"imgsrclist": imgsrclist}


@app.get("/pdfcontents/", tags=["url"], response_model=list[str])
async def get_pdf_contents(url: str):
    # ユニークな一時ファイル名を自動生成させる。
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        file_name = tmp_file.name
        try:
            urlretrieve(url, file_name)
            reader = PdfReader(file_name)
            return [page.extract_text() for page in reader.pages]
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)


@app.get("/tokenizedwords/", tags=["text"], response_model=list[str])
async def get_tokenized_words(text: str):
    tokens = word_tokenize(text)
    return Text(tokens)


@app.get("/nouninsentences/", tags=["text"], response_model=dict[str, list[str]])
async def get_noun_in_sentences(
    sentences: list[str] = Query(
        ..., example=["My name is Taro", "My students are studying hard"]
    ),
):
    """
    文の中の名詞を抽出します。
    sentences: strのリストで、各文を含む。
    戻り値は、名詞の品詞タグとその名詞のリストを含む辞書。
    """
    nouns_tags = ["NN", "NNS", "NNP", "NNPS"]
    result = {}
    for sentence in sentences:
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag in nouns_tags:
                result.setdefault(tag, []).append(word)

    return result


async def to_pil_image(image):
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents))
    return pil_image


def extract_text_from_image(pil_image):
    """
    引数のPIL Imggeからテキストを抽出します。現時点では日本語のテキストを対象とします。
    """
    return pytesseract.image_to_string(pil_image, lang="jpn")


@app.post("/imagetext/", tags=["image"], response_model=dict[str, str])
async def get_text_in_image(image: UploadFile = File(...)):
    """
    アップロードされた画像からテキストを抽出します。
    """
    try:
        pil_image = to_pil_image(image)
    except Exception as e:
        return {"error": f"画像の読み込みに失敗しました: {e}"}

    return {"text": extract_text_from_image(pil_image)}


@app.get("/imageurltext/", tags=["url"], response_model=dict[str, str])
async def get_text_in_image_url(
    url: str = Query(
        ...,
        example="https://asset.watch.impress.co.jp/img/ipw/docs/2039/347/open1_o.jpg",
    ),
):
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp_file:
        file_name = tmp_file.name
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error: Content-Type '{content_type}' is not an image."
                )

            pil_image = Image.open(BytesIO(response.content))
            return {"text": extract_text_from_image(pil_image)}
        except Exception as e:            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e.args[0]
            )
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)
