from fastapi.testclient import TestClient
from webscraping import app
import pytest
import os


def test_get_hellogis():
    with TestClient(app) as test_client:
        response = test_client.get("/hellowebscraping/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello Brest Web Scraping!"}


def test_get_page_contents():
    with TestClient(app) as test_client:
        response = test_client.get("/pagecontents/?url=http://localhost/")
        assert response.status_code == 200
        contents = response.json()["contents"]
        assert contents is not None


def test_error_http_exception_when_not_found_title():
    with TestClient(app) as test_client:
        """存在しないページのタイトルを得ようとした時に500エラーになることのテストです。"""
        response = test_client.get("/pagetitle/?url=http://localhost/errorsite")
        assert response.status_code == 400


def test_get_page_image_list():
    with TestClient(app) as test_client:
        response = test_client.get(
            "/pageimgsrclist/?url=http://localhost/webxam/css/multicolumnlayout/"
        )
        assert response.status_code == 200
        srclist = response.json()["imgsrclist"]
        assert srclist is not None
        assert len(srclist) > 0


def test_get_page_png_image_list():
    with TestClient(app) as test_client:
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
    with TestClient(app) as test_client:
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
    with TestClient(app) as test_client:
        text = "I have a pen"
        response = test_client.get(f"/tokenizedwords/?text={text}")
        assert response.status_code == 200
        words = response.json()
        assert words is not None
        assert len(words) == 4


def test_get_noun_in_sentences():
    """
    文の中の名詞を抽出するAPIのテストです。
    """
    with TestClient(app) as test_client:
        sentences = ["My name is Taro", "My students are studying hard"]
        response = test_client.get("/nouninsentences/", params={"sentences": sentences})
        assert response.status_code == 200
        result = response.json()
        assert result is not None
        assert len(result) > 0
        print(result)


def test_get_text_in_image():
    with TestClient(app) as test_client:
        # プロジェクトのルートディレクトリ以下にあるsampleディレクトリからテスト用画像を読み込む
        image_path = os.path.join(
            os.path.dirname(__file__), ".", "sample", "sample_text_image.png"
        )

        with open(image_path, "rb") as image_file:
            # 'files'パラメータを使ってUploadFileとして送信します
            # ファイルデータはバイトデータとして直接渡します
            files = {"image": ("sample_text_image.png", image_file, "image/png")}
            response = test_client.post("/imagetext/", files=files)

        assert response.status_code == 200

        assert "text" in response.json()

        expected_text = response.json()["text"]
        print(expected_text)
        assert expected_text is not None

        # 抽出されたテキスト群は改行で区切られる。
        assert "毎日暑くて体に厳しい" in expected_text
        # サロゲートペア文字を含む場合、正しく抽出できていない。
        # 似た別の文字に置き換えてくれる場合もあるが「𩸽」は全く別の文字になってしまう。
        # 改行の後かそうでないかによってどの文字になってしまうか変化する。
        assert "𩸽" in expected_text
        assert "𠮟" in expected_text
        assert "𠮷" in expected_text


def test_get_text_from_image_url():
    """
    画像URLから画像のテキストを読み込むAPIのテストです。
    """
    with TestClient(app) as test_client:
        url = "https://asset.watch.impress.co.jp/img/ipw/docs/2039/347/open1_o.jpg"
        response = test_client.get("/imageurltext/", params={"url": url})
        assert response.status_code == 200
        assert "text" in response.json()

        result = response.json()["text"]
        assert result is not None

        assert len(result) > 0
        print(result)
