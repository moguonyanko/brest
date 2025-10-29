"""WebスクレイピングのAPIを提供します。

参考資料:
PythonによるWebスクレイピング 第3版
https://www.oreilly.co.jp/books/9784814401222/
"""

import re
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException, status, Query
from fastapi import File, UploadFile
import os
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError
from bs4 import BeautifulSoup
from pypdf import PdfReader
import tempfile
from nltk import word_tokenize, Text, pos_tag
from PIL import Image
import pytesseract
import httpx
from playwright.async_api import async_playwright, Page
from genaiapi import get_genai_client
import ssl

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


_CADDY_ROOT_CERT_PATH = os.environ["CADDY_ROOT_CERT_PATH"]


def _create_custom_ssl_context(cert_path: str):
    """Caddyのルート証明書を追加したSSLコンテキストを作成"""
    if not os.path.exists(cert_path):
        raise FileNotFoundError(f"証明書ファイルが見つかりません: {cert_path}")

    print(f"DEBUG: Caddyルート証明書をロード中: {cert_path}")
    context = ssl.create_default_context()
    # Caddyのルート証明書を信頼されたCAファイルとして追加
    context.load_verify_locations(cafile=cert_path)
    return context


custom_ssl_context = _create_custom_ssl_context(_CADDY_ROOT_CERT_PATH)


def read_all_contents(url: str) -> str:
    try:
        contents = urlopen(url, context=custom_ssl_context)
    except URLError as e:
        # e.reasonがオブジェクトの場合があるため、e全体を文字列化する
        detail_msg = str(e)
        # エラーがSSLCertVerificationErrorであることを示すため、詳細情報に含める
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            detail_msg = f"SSL証明書エラー: {str(e.reason)}"
        else:
            detail_msg = str(e)

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail_msg)

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
        ..., examples=["My name is Taro", "My students are studying hard"]
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
        pil_image = await to_pil_image(image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"画像の読み込みに失敗しました: {e}",
        )

    return {"text": extract_text_from_image(pil_image)}


@app.get("/imageurltext/", tags=["url"], response_model=dict[str, str])
async def get_text_in_image_url(
    url: str = Query(
        ...,
        examples="https://asset.watch.impress.co.jp/img/ipw/docs/2039/347/open1_o.jpg",
    ),
):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error: Content-Type '{content_type}' is not an image.",
            )

        pil_image = Image.open(BytesIO(response.content))
        return {"text": extract_text_from_image(pil_image)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def _search_shops_by_station(page: Page):
    station_selector = "#sectionSideSearchBox > div > div.s_side_search_keyword.s_side_search_keyword_area > form > div > div > div > input"
    await page.wait_for_selector(station_selector, state="visible")
    await page.fill(station_selector, "浜松町駅")

    search_button = "#sectionSideSearchBox > div > div.s_side_search_keyword.s_side_search_keyword_area > form > div > input"
    await page.wait_for_selector(search_button, state="visible")
    await page.click(search_button)


async def _select_station(page: Page):
    target_station_link = page.get_by_text("浜松町駅（ＪＲ山手線）")
    await target_station_link.wait_for(state="visible")
    await target_station_link.click()


async def _extract_supermarkets(page: Page):
    sort_selector = "#main-content-inner > div.content_main_section.section_box_type_E > div > ul > li.chirashi_list_option_sort > span > a"
    await page.wait_for_selector(sort_selector, state="visible")
    await page.click(sort_selector)

    near_distance_selector = f"#pop_menu_2 > div > div > ul > li:nth-child(1) > a"
    await page.wait_for_selector(near_distance_selector, state="visible")
    await page.click(near_distance_selector)
    # get_by_textは動作が安定しない。
    # sort_element = page.locator(sort_selector)
    # near_distance_element = sort_element.get_by_text("近い順", exact=True)
    # await near_distance_element.click()

    shop_type_selector = "#main-content-inner > div.content_main_section.section_box_type_E > div > ul > li.chirashi_list_option_filters.single_option.btn_sp_ui.btn_sp_ui_A > a"
    await page.wait_for_selector(shop_type_selector, state="visible")
    await page.click(shop_type_selector)

    all_type_link = "#main-content-inner > div.content_main_section.section_box_type_E > div > ul > li.chirashi_list_option_filters.single_option.btn_sp_ui.btn_sp_ui_A > a"
    await page.wait_for_selector(all_type_link, state="visible")
    await page.click(all_type_link)

    super_link_selector = (
        "#pop_menu_1 > div > div > div > ul > li:nth-child(2) > a > span"
    )
    await page.wait_for_selector(super_link_selector, state="visible")
    await page.click(super_link_selector)


async def _get_chirashi_image(page: Page):
    element = page.locator("#chirashi-area")
    image_bytes = await element.screenshot()
    image = Image.open(BytesIO(image_bytes))
    return image


async def _get_chirashi_data(page: Page):
    """
    チラシの内容を返します。
    """
    chirashi_data = {}
    client = get_genai_client()
    upload_images = []
    shop_names = []

    chirashi_list = await page.locator(".chirashi_list_item").all()
    for chirashi_item in chirashi_list:
        await chirashi_item.wait_for(state="visible")
        shop_name = await chirashi_item.locator(
            ".chirashi_list_item_name_str"
        ).text_content()
        await chirashi_item.click()
        chirashi_image = await _get_chirashi_image(page)
        upload_image = await client.aio.files.upload(file=chirashi_image)
        upload_images.append(upload_image)
        shop_names.append(shop_name)
        await page.go_back()

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            *upload_images,
            "Please extract the text from the image.",
        ],
    )

    for target_shop_name in shop_names:
        chirashi_data[target_shop_name] = response.text

    return chirashi_data


@app.get("/tokubai/", tags=["url"], response_model=dict[str, str])
async def get_tokubai_info():
    target_url = "https://www.shufoo.net/"
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            # テスト用
            # headless=False,
            # slow_mo=1000,
        )
        page = await browser.new_page()

        try:
            await page.goto(target_url, wait_until="domcontentloaded")

            await _search_shops_by_station(page)

            await _select_station(page)

            await _extract_supermarkets(page)

            chirashi_list = await _get_chirashi_data(page)

            return chirashi_list
        except Exception as e:
            print(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
            )
        finally:
            await page.screenshot(
                path="dist/debug_screenshot_before_super_click_final.png"
            )
            # ブラウザを閉じる
            await browser.close()
