from fastapi.testclient import TestClient
from webscraping import app
import pytest

test_client = TestClient(app)


def test_get_hellogis():
    response = test_client.get("/hellowebscraping/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Brest Web Scraping!"}


def test_get_page_contents():
    response = test_client.get("/pagecontents/?url=http://localhost/")
    assert response.status_code == 200
    contents = response.json()["contents"]
    assert contents is not None


def test_error_http_exception_when_not_found_title():
    """存在しないページのタイトルを得ようとした時に500エラーになることのテストです。"""
    response = test_client.get("/pagetitle/?url=http://localhost/errorsite")
    assert response.status_code == 400


def test_get_page_image_list():
    response = test_client.get(
        "/pageimgsrclist/?url=http://localhost/webxam/css/multicolumnlayout/"
    )
    assert response.status_code == 200
    srclist = response.json()["imgsrclist"]
    assert srclist is not None
    assert len(srclist) > 0


def test_get_page_png_image_list():
    format = "png"
    response = test_client.get(
        f"/pageimgsrclist/?url=http://localhost/webxam/css/multicolumnlayout/&format={format}"
    )
    assert response.status_code == 200
    srclist = response.json()["imgsrclist"]
    assert srclist is not None
    assert len(srclist) > 0
    for src in srclist:
        assert src.endswith(format)


def test_get_pdf_contents():
    # sample_url = "https://rirs.or.jp/tenken-db/pdf/api_specification.pdf"
    sample_url = "http://localhost/webxam/document/samplebook.pdf"
    response = test_client.get(f"/pdfcontents/?url={sample_url}")
    assert response.status_code == 200
    pages = response.json()
    assert pages is not None
    assert len(pages) > 0
    for page in pages:
        print(page)


def test_get_tokenized_words():
    # TODO: 日本語の分割はできない？
    text = "私は甘いお菓子が好きです"
    # text = "彼は𩸽を𠮟る𠮷野家に勤めている"
    # 英語の文章は分割できる。
    # text = "I have a pen"
    response = test_client.get(f"/tokenizedwords/?text={text}")
    assert response.status_code == 200
    words = response.json()
    assert words is not None
    assert len(words) > 1
