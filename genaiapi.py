import requests
import time
from io import BytesIO
from typing import Annotated, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, UrlContext
from PIL import Image
from fastapi import FastAPI, HTTPException, Body, Response, Form
from fastapi.responses import StreamingResponse, FileResponse
from starlette.background import BackgroundTask
from fastapi import File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from urllib.parse import quote
import wave
from pathlib import Path
import uuid
import os
from utils import (
    load_json,
    convert_normalized_bbox_to_pixel_bbox,
    Point,
    convert_normalized_point_to_pixel_point,
)
from genaiappi_models_utils import (
    get_generate_text_model_name,
    get_generate_image_model_name,
    get_generate_vision_model_name,
    get_generate_transcription_movie_model_name,
    get_generate_transcription_movie_inline_model_name,
    get_generate_transcription_audio_inline_model_name,
    get_model_name_summarize_document,
    get_model_name_text_embedding,
    get_model_name_thinking,
    get_model_generate_speech,
    get_model_url_context,
    get_model_live_api_speech,
    get_model_robotics,
)
import json

app = FastAPI(
    title="Brest Generative AI API",
    description="生成AIを操作する機能をAPIで提供する。",
    summary="Gen AI API by REST",
    version="0.0.1",
)

app_base_path = "/generate"


# Gemini APIの共通設定読み込み
genaiapi_config = load_json(path="genaiapi_config.json")


# Gemini APIのキー読み込み
api_keys = load_json(path="genaiapi_keys.json")["api_keys"]
# APIキーを使い分ける必要が生じたらcommon以外を参照できるように以下のコードを修正する。
api_key = api_keys["common"]


def get_genai_client():
    timeout_ms = genaiapi_config["resuest_timeout_ms"]
    return genai.Client(api_key=api_key, http_options={"timeout": timeout_ms})


"""
生成結果をJSONの形式で返すためのクラス
"""


class GenerationResultText(BaseModel):
    text: str


@app.post(f"{app_base_path}/text/", tags=["ai"], response_model=GenerationResultText)
async def generate_text(body: dict):
    response = get_genai_client().models.generate_content(
        model=get_generate_text_model_name(),
        contents=body["contents"],
        config={
            "response_mime_type": "application/json",
            "response_schema": GenerationResultText,
        },
    )
    return response.parsed


@app.post(
    f"{app_base_path}/text-from-image/",
    tags=["ai"],
    response_model=GenerationResultText,
)
async def generate_text_from_image(
    file: Annotated[UploadFile, File(description="プロンプトに渡す画像です。")],
):
    try:
        request_file_content = await file.read()
        image = Image.open(BytesIO(request_file_content))
        response = get_genai_client().models.generate_content(
            model=get_generate_text_model_name(),
            contents=[image, "画像について説明してください。"],
            config={
                "response_mime_type": "application/json",
                "response_schema": GenerationResultText,
            },
        )
        # 回答は英語で返されてしまう。
        return response.parsed
    except Exception:
        raise HTTPException(
            status_code=500, detail="画像による問い合わせに失敗しました。"
        )
    finally:
        file.file.close()
        image.close()


async def close_websocket(websocket: WebSocket):
    try:
        await websocket.close()
        print(f"WebSocket connection closed for {websocket.client}")
    except RuntimeError as e:
        # 'Unexpected ASGI message' エラーは無視する（既にclose済みとみなす）
        if "Unexpected ASGI message 'websocket.close'" in str(e):
            print(
                f"WebSocket connection already closed (ignoring error): {websocket.client}"
            )
        else:
            # その他の RuntimeError はログ出力しておく
            print(f"RuntimeError during websocket.close(): {e}")
    except Exception as e:
        print(f"Error during websocket.close(): {e}")


@app.websocket(f"{app_base_path}/talk/")
async def talk_generative_ai(websocket: WebSocket):
    await websocket.accept()
    chat = get_genai_client().chats.create(
        model=get_generate_text_model_name(),
        config=types.GenerateContentConfig(
            max_output_tokens=genaiapi_config["max_output_tokens"],
            temperature=genaiapi_config["temperature"],
        ),
    )
    try:
        while True:
            user_message = await websocket.receive_text()
            response = chat.send_message(user_message)
            await websocket.send_text(response.text)
    except WebSocketDisconnect as disconnect_err:  # ブラウザでcloseした。
        print(f"Client disconnected from {websocket.client}, detail={disconnect_err=}")
    except RuntimeError as runtime_err:
        print(f"WebSocket runtime error: {runtime_err=}")
    finally:
        # WebSocketDisconnect以外のエラー発生時はWebSocketが閉じられていない可能性があるので
        # 念のためWebSocketを閉じる関数を呼びなおす。
        await close_websocket(websocket)


@app.post(
    f"{app_base_path}/image/",
    tags=["ai"],
    responses={200: {"content": {"image/*": {}}}, 500: {"content": {"text/plain": {}}}},
    response_class=Response,
)
async def generate_image(
    body: Annotated[
        dict, Body(examples=[{"contents": "Generate an image of an apple."}])
    ],
):
    """
    リクエストされたテキストから画像を生成します。
    """
    try:
        response = get_genai_client().models.generate_content(
            model=get_generate_image_model_name(),
            contents=body["contents"],
            config=types.GenerateContentConfig(
                response_modalities=[
                    "Text",
                    "Image",
                ]  # Textは指定されていないとエラーになる。
            ),
        )

        # 最初のcandidatesだけ返している。
        image_text = ""
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                # ここでは単に出力しているだけだがWebSocketを使えば作成前のメッセージを返しつつ
                # 画像も返すことができそうである。ここではカスタムHTTPヘッダーを使って
                # 画像の説明を返している。
                # HTTPヘッダーの値はlatin-1が想定されているためquoteでエンコードしないとエラーになる。
                image_text = quote(part.text)
                # return Response(content=part.text, media_type='text/plain')
            elif part.inline_data is not None:
                image_bytes = BytesIO((part.inline_data.data))
                image_bytes.seek(0)
                return Response(
                    content=image_bytes.getvalue(),
                    media_type=part.inline_data.mime_type,
                    headers={"X-Generation-Image-Text": image_text},
                )
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Generative Error: {err=}, {type(err)=}"
        )


@app.post(
    f"{app_base_path}/text-from-image-url/",
    tags=["ai"],
    response_model=GenerationResultText,
)
async def generate_text_from_image_url(body: dict):
    try:
        image = requests.get(body["url"])
        content_type = image.headers.get("Content-Type")
        response = get_genai_client().models.generate_content(
            model=get_generate_vision_model_name(),
            contents=[
                "画像について説明してください。",
                types.Part.from_bytes(data=image.content, mime_type=content_type),
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": GenerationResultText,
            },
        )
        # 回答は英語になってしまう。
        return response.parsed
    except Exception:
        raise HTTPException(
            status_code=500, detail="画像による問い合わせに失敗しました。"
        )


"""
参考:
https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#bbox
"""


@app.post(f"{app_base_path}/bouding-box-from-image/", tags=["ai"], response_model=str)
async def generate_bounding_box_from_image(
    file: Annotated[UploadFile, File(description="プロンプトに渡す画像です。")],
):
    try:
        request_file_content = await file.read()
        image = Image.open(BytesIO(request_file_content))
        prompt = (
            "Return a bounding box for each of the objects in this image "
            "in [ymin, xmin, ymax, xmax] format."
        )
        response = get_genai_client().models.generate_content(
            model=get_generate_text_model_name(),
            contents=[image, prompt],
            config={"response_mime_type": "application/json"},
        )
        return response.text
    except Exception:
        raise HTTPException(
            status_code=500, detail="画像による問い合わせに失敗しました。"
        )
    finally:
        file.file.close()
        image.close()


"""
参考:
https://ai.google.dev/gemini-api/docs/vision?hl=ja&lang=python#prompt-video
"""


@app.post(f"{app_base_path}/transcription-from-movie/", tags=["ai"], response_model=str)
async def generate_transcription_from_movie(
    file: Annotated[UploadFile, File(description="プロンプトに渡す動画です。")],
):
    try:
        client = get_genai_client()
        video_file = client.files.upload(
            file=file.file, config={"mime_type": file.content_type}
        )

        # アップロードされた動画が利用可能になるまで待機する。
        while video_file.state.name == "PROCESSING":
            print(".", end="")
            time.sleep(1)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        prompt_for_movie_summary = (
            "Please provide the Japanese text you would like me to translate into English."
            "Once you provide the Japanese text, I will also generate a summary of its content."
        )

        response = client.models.generate_content(
            model=get_generate_transcription_movie_model_name(),
            contents=[video_file, prompt_for_movie_summary],
            config=types.GenerateContentConfig(**genaiapi_config),
        )
        return response.text
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Prompt Error: {err=}, {type(err)=}"
        )


"""
動画をインラインでアップロードして要約を生成します。
"""


@app.post(
    f"{app_base_path}/transcription-inline-from-movie/", tags=["ai"], response_model=str
)
async def generate_transcription_inline_from_movie(
    file: Annotated[UploadFile, File(description="プロンプトに渡す動画です。")],
):
    try:
        client = get_genai_client()

        prompt_for_movie_summary = (
            "Please provide the Japanese text you would like me to translate into English."
            "Once you provide the Japanese text, I will also generate a summary of its content."
        )

        video_bytes = await file.read()

        response = client.models.generate_content(
            model=get_generate_transcription_movie_inline_model_name(),
            contents=types.Content(
                parts=[
                    types.Part(text=prompt_for_movie_summary),
                    types.Part(
                        inline_data=types.Blob(
                            data=video_bytes, mime_type=file.content_type
                        )
                    ),
                ]
            ),
            config=types.GenerateContentConfig(**genaiapi_config),
        )
        return response.text
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Generation Error: {err=}, {type(err)=}"
        )


"""
音声ファイルをインラインでAPIに渡して文字起こしかつ要約します。
"""


@app.post(
    f"{app_base_path}/transcription-inline-from-audio/", tags=["ai"], response_model=str
)
async def generate_transcription_inline_from_auido(
    file: Annotated[UploadFile, File(description="プロンプトに渡す音声です。")],
):
    client = get_genai_client()

    prompt_for_audio_summary = (
        "Please transcribe the audio file into Japanese and then summarize it."
    )

    audio_bytes = await file.read()

    try:
        response = client.models.generate_content(
            model=get_generate_transcription_audio_inline_model_name(),
            contents=[
                prompt_for_audio_summary,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=file.content_type,
                ),
            ],
            config=types.GenerateContentConfig(**genaiapi_config),
        )
        return response.text
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Generation Error: {err=}, {type(err)=}"
        )


@app.post(
    f"{app_base_path}/summarization-from-document/",
    tags=["ai"],
    response_model=str,
    description="アップロードされたドキュメントから要約を生成します。",
)
async def generate_summarization_from_document(
    file: Annotated[UploadFile, File(description="処理対象ドキュメントです。")],
):
    try:
        # ドキュメントが大きい場合は読み込み方法を再考する。
        doc_bytes = await file.read()

        prompt = "文書の要約を生成してください"
        response = get_genai_client().models.generate_content(
            model=get_model_name_summarize_document(),
            contents=[
                types.Part.from_bytes(
                    data=doc_bytes,
                    mime_type=file.content_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(**genaiapi_config),
        )

        return response.text
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Generation Error: {err=}, {type(err)=}"
        )


def calculate_pairwise_similarities(embeddings: list):
    """
    複数の埋め込みベクトル間のペアワイズなコサイン類似度を計算します。

    Args:
        embeddings (list of list or numpy.ndarray): 埋め込みベクトルのリスト。

    Returns:
        list of float: すべてのペアのコサイン類似度のリスト。
    """
    num_embeddings = len(embeddings)
    similarities = []
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            emb1 = np.array(embeddings[i].values).reshape(1, -1)
            emb2 = np.array(embeddings[j].values).reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            similarities.append(similarity)
    return similarities


def aggregate_similarity(pairwise_similarities, aggregation_method="mean"):
    """
    ペアワイズな類似度を特定の方法で集約します。

    Args:
        pairwise_similarities (list of float): ペアワイズな類似度のリスト。
        aggregation_method (str): 集約方法 ("mean", "min", "median"のいずれか)。
        デフォルトは "mean"。

    Returns:
        float: 集約された類似度。
    """
    if not pairwise_similarities:
        return 0.0

    if aggregation_method == "mean":
        return np.mean(pairwise_similarities)
    elif aggregation_method == "min":
        return np.min(pairwise_similarities)
    elif aggregation_method == "median":
        return np.median(pairwise_similarities)
    else:
        raise ValueError("'mean', 'min', 'median' のいずれかを指定してください。")


@app.post(
    f"{app_base_path}/text-similarity/",
    tags=["ai"],
    response_model=float,
    description="テキストの埋め込みを利用してテキストの類似性を取得します。",
)
async def generate_text_similarity(
    body: Annotated[
        dict, Body(examples=[{"contents": ["Hello, World", "こんにちは、世界"]}])
    ],
):
    try:
        contents = body["contents"]

        response = get_genai_client().models.embed_content(
            model=get_model_name_text_embedding(),
            contents=contents,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )

        pairwise_similarities = calculate_pairwise_similarities(response.embeddings)
        return aggregate_similarity(pairwise_similarities)
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Generation Error: {err=}, {type(err)=}"
        )


@app.post(
    f"{app_base_path}/price-prediction",
    tags=["ai"],
    response_model=dict,
    description="ある商品が指定された期間の間にどの程度の価格になるのか推定します。",
)
async def generate_price_prediction(
    body: Annotated[
        dict,
        Body(examples=[{"contents": {"name": "米", "period": "3ヶ月", "unit": "円"}}]),
    ],
):
    """
    ある商品が指定された期間の間にどの程度の価格になるのか推定します。
    """
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    contents = body["contents"]
    item = contents["name"]
    period = contents["period"]
    unit = contents["unit"]

    prompt = f"{item}が{period}経過したら何{unit}になるのか価格を推測してください"

    grounding_config = {"tools": [google_search_tool], "response_modalities": ["TEXT"]}
    copied_config = genaiapi_config.copy()
    copied_config.update(grounding_config)

    response = get_genai_client().models.generate_content(
        model=get_generate_text_model_name(),
        contents=prompt,
        config=types.GenerateContentConfig(**copied_config),
    )

    # rendered_contentには関連情報をGoogle検索するためのHTML要素が保持されている。
    # if response.candidates[0].grounding_metadata.search_entry_point is not None:
    #     print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)

    text_list = []
    for each in response.candidates[0].content.parts:
        text_list.append(each.text)

    return {"result": text_list}


@app.post(
    f"{app_base_path}/travel-project",
    tags=["ai"],
    response_model=str,
    description="旅行計画を立案します。",
)
async def generate_travel_project(
    body: Annotated[
        dict,
        Body(
            examples=[
                {
                    "contents": {
                        "position": {"start": "横浜", "end": "仙台"},
                        "period": {"start": "2025/05/5", "end": "2025/05/7"},
                        "purpose": "美味しいものが食べたい,かえるのピクルスのグッズを買いたい,有名な建物を見に行きたい",
                    }
                }
            ]
        ),
    ],
):
    contents = body["contents"]
    position = contents["position"]
    period = contents["period"]
    purpose = contents["purpose"]
    prompt = f"""
    旅行計画を立案してください。
    出発地は{position['start']}で目的地は{position['end']}です。
    日程は{period['start']}から{period['end']}です。
    目的は「{purpose}」です。
    """
    prompt = prompt.replace("\n", "")

    # グラウンディングを利用して回答を補強する。
    grounding_config = {"tools": [types.Tool(google_search=types.GoogleSearch())]}
    copied_config = genaiapi_config.copy()
    copied_config.update(grounding_config)

    response = get_genai_client().models.generate_content(
        model=get_model_name_thinking(),
        contents=prompt,
        # thinking_budgetを指定するとValidationErrorになる。
        config=types.GenerateContentConfig(**copied_config),
    )

    return response.text


def write_wave_file(file_path: Path, frames, channels=1, rate=24000, sample_width=2):
    """
    生成された音声データをローカルで確認できるようにするために書き出す関数です。
    """
    with wave.open(os.fspath(file_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(frames)


def dump_base64_data_type(data):
    print(f"Type of data: {type(data)}")
    print(f"Length of data: {len(data)}")
    # dataが文字列の場合、最初の50文字程度を表示（長すぎると読みにくいため一部だけ出力）
    if isinstance(data, str):
        print(f"First 50 characters of data: {data[:50]}...")
    # dataがバイト列の場合、最初の50バイト程度をHex形式で表示
    elif isinstance(data, bytes):
        print(f"First 50 bytes of data (hex): {data[:50].hex()}...")


def get_temp_audio_file_path(prefix: str = "") -> Path:
    temp_dir = Path(f"{Path.home()}/share/audio/tmp/fastapi_audio_cache")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / f"{prefix}_speech_{uuid.uuid4()}.wav"
    return temp_file_path


def write_temp_wave_file(frames) -> Path:
    """
    一時的なwaveファイルを書き出す関数です。
    FileResponseを使うために作られました。
    """
    temp_file_path = get_temp_audio_file_path()
    write_wave_file(temp_file_path, frames)
    return temp_file_path


def remove_file(file_path_name: str):
    os.unlink(file_path_name)


@app.post(
    f"{app_base_path}/speech-generation/",
    tags=["ai"],
    description="アップロードされたドキュメントから音声を生成します。",
)
async def generate_speech_from_document(
    file: Annotated[UploadFile, File(description="処理対象のドキュメント")],
):
    """
    StreamingResponseはPydanticモデルではないためresponse_modelに指定するとFastAPIErrorとなる。
    StreamingResponseを使うと空の音声データがクライアントに返されてしまうためFileResponseを使用している。
    """
    doc_bytes = await file.read()
    contents = doc_bytes.decode("utf-8")

    response = get_genai_client().models.generate_content(
        model=get_model_generate_speech(),
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
        ),
    )

    frames = response.candidates[0].content.parts[0].inline_data.data
    dump_base64_data_type(frames)

    if frames is None or len(frames) == 0:
        raise HTTPException(
            status_code=500, detail="音声データが生成できませんでした。"
        )

    temp_file_path = write_temp_wave_file(frames)

    return FileResponse(
        path=temp_file_path,
        media_type="audio/wav",
        filename="generated_speech.wav",
        background=BackgroundTask(remove_file, temp_file_path),
    )


@app.post(f"{app_base_path}/inspect-url-context/", tags=["ai"], response_model=str)
async def inspect_url_context(
    body: Annotated[
        dict,
        Body(
            examples=[
                {
                    "contents": [
                        {
                            "url": "https://tenki.jp/",
                            "operation": "本日の全国の天気を要約してください。",
                        }
                    ]
                }
            ]
        ),
    ],
):
    client = get_genai_client()
    model_id = get_model_url_context()
    contents = body["contents"][0]  # 現状は最初の要素しか扱わない。
    target_url = contents["url"]
    operation = contents["operation"]

    tools = []
    tools.append(Tool(url_context=UrlContext))
    tools.append(Tool(google_search=GoogleSearch))

    response = client.models.generate_content(
        model=model_id,
        contents=f"{target_url}を調べて次の処理を行ってください。{operation}",
        config=GenerateContentConfig(tools=tools, response_modalities=["TEXT"]),
    )

    result_text = []
    # 各textをデバッグしやすくするためリスト内包表記を用いていない。
    for each in response.candidates[0].content.parts:
        result_text.append(each.text)

    return " ".join(result_text)


@app.post(f"{app_base_path}/grounding-text/", tags=["ai"], response_model=str)
async def generate_text_with_googlesearch_grounding(
    body: Annotated[
        dict, Body(examples=[{"contents": "今日の東京の天気を教えてください。"}])
    ],
):
    """
    Google検索を利用して、指定されたテキストに基づいて情報を取得し、回答を生成します。
    このAPIは、Google検索をツールとして使用し、指定されたテキストを基に情報を検索し、回答を生成します。
    ツールの使用によって、最新の情報を取得し、より正確な回答を提供することが可能です。
    例えば、天気予報や最新のニュースなど、リアルタイムで変化する情報を取得するのに適しています。
    """
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    response = get_genai_client().models.generate_content(
        model=get_generate_text_model_name(),
        contents=body["contents"],
        config=GenerateContentConfig(tools=[google_search_tool]),
    )
    return response.text


@app.post(f"{app_base_path}/live-api/text-to-speech", tags=["ai"])
async def generate_speech_from_text_by_live_api(
    body: Annotated[dict, Body(examples=[{"contents": "自己紹介をしてください。"}])],
):
    """
    リクエストされたテキストから音声を生成します。
    このAPIは、GeminiのライブAPIを使用して、リアルタイムでテキストから音声を生成します。
    リクエストボディには、生成したいテキストを含む`contents`フィールドが必要です。
    音声データは一時的なファイルに保存され、レスポンスとして返されます。
    音声ファイルはWAV形式で、サンプリングレートは24000Hz、チャンネル数は1（モノラル）で生成されます。
    """
    client = get_genai_client()
    model = get_model_live_api_speech()
    config = {"response_modalities": ["AUDIO"]}
    temp_file_path = get_temp_audio_file_path(prefix="text-to-speech")

    async with client.aio.live.connect(model=model, config=config) as session:
        channels = 1
        sampwidth = 2
        framerate = 24000
        wf = wave.open(os.fspath(temp_file_path), "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)

        message = body["contents"]
        await session.send_client_content(
            turns={"role": "user", "parts": [{"text": message}]}, turn_complete=True
        )

        async for response in session.receive():
            if response.data is not None:
                wf.writeframes(response.data)

        wf.close()

    return FileResponse(
        path=temp_file_path,
        media_type="audio/wav",
        filename="generated_text_to_speech.wav",
        background=BackgroundTask(remove_file, temp_file_path),
    )


def _normalize_bounding_box(result: list[Any], image_bytes: bytes):
    """
    画像のバウンディングボックスを正規化します。
    引数のresultはサイズが大きい可能性がありコピーを避けるため、
    resultの値を上書きして返します。従ってこの関数は副作用があります。
    なお、resultはAPIの戻り値をそのまま渡すことを想定しています。
    そのため、戻り値の型はlist[Any]としています。
    画像のバウンディングボックスは[ ymin, xmin, ymax, xmax ]形式で並んでいることを想定しています。
    """
    original_image = Image.open(BytesIO(image_bytes))
    for bounding_box_info in result:
        normalized_box = convert_normalized_bbox_to_pixel_bbox(
            bounding_box=bounding_box_info["bounding_box"],
            original_image_size=(original_image.width, original_image.height),
            scale=genaiapi_config["normalization_scale"],
        )
        bounding_box_info["bounding_box"] = normalized_box


def _request_robotics_api(
    file: UploadFile, image_bytes: bytes, prompt: str
) -> dict[str, Any]:
    client = get_genai_client()

    image_response = client.models.generate_content(
        model=get_model_robotics(),
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=genaiapi_config["temperature"],
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        ),
    )

    return json.loads(image_response.text)


@app.post(
    f"{app_base_path}/robotics/detect-objects",
    tags=["ai"],
    response_model=dict[str, Any],
)
async def detect_objects(
    files: Annotated[
        list[UploadFile], File(description="アップロードされたファイル群です。")
    ],
    # 内容はリストだがJSONの文字列として渡されてくるのでstr型で受け取る。
    targets: Annotated[
        str,
        Form(
            description="検出対象の名前を保持した文字列のリストです。",
            examples=['{"targets": ["apple"]}'],
        ),
    ],
):
    """
    ロボティクス用モデルを使って画像内のオブジェクトを検出します。
    """
    try:
        targets_list = json.loads(targets)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="targetsパラメータのJSONデコードに失敗しました。有効なJSON文字列を指定してください。",
        )

    prompt = f"""
    次の画像に含まれるオブジェクトを検出してください。検出対象は{targets_list}です。
    検出結果はJSON形式で、各オブジェクトの種類と バウンディングボックスの座標を含めてください。
    バウンディングボックスの座標は[ymin, xmin, ymax, xmax]形式で指定してください。
    画像に検出対象が存在しない場合は空のリストを返してください。
    # 出力例:
    [
        {{"object": "human", "bounding_box": [431, 220, 512, 296]}},
        {{"object": "car", "bounding_box": [459, 642, 539, 719]}}
    ]
    """
    results = {}

    for file in files:
        try:
            image_bytes = await file.read()
            result = _request_robotics_api(
                file=file,
                image_bytes=image_bytes,
                prompt=prompt,
            )

            _normalize_bounding_box(result, image_bytes)

            results[file.filename] = result
        except Exception as err:
            code = hasattr(err, "status_code") and getattr(err, "status_code")
            if not code:
                code = 500

            raise HTTPException(
                status_code=code,
                detail=f"画像処理エラー: ファイル {file.filename} の処理中にエラーが発生しました。{err=}, {type(err)=}",
            )

    return results


def _normalized_point_to_pixel_point(
    point_info_response: dict[str, Any], image_bytes: bytes
):
    """
    正規化された座標をピクセル座標に変換します。この関数はpoint_info_responseに対して副作用があります。
    画像のバウンディングボックスは[ y, x ]形式で並んでいることを想定しています。
    なお、point_info_responseはAPIの戻り値をそのまま渡すことを想定しています。
    """
    for point_info in point_info_response:
        normalized_point = Point(
            y=point_info["point"][0],
            x=point_info["point"][1],
        )

        (width, height) = Image.open(BytesIO(image_bytes)).size

        pixel_point = convert_normalized_point_to_pixel_point(
            normalized_point=normalized_point,
            original_image_size=(width, height),
            scale=1000,
        )

        point_info["point"] = [pixel_point.y, pixel_point.x]


@app.post(
    f"{app_base_path}/robotics/task-orchestration",
    tags=["ai"],
    response_model=dict[str, Any],
)
async def get_task_orchestration(
    files: Annotated[
        list[UploadFile], File(description="アップロードされたファイル群です。")
    ],
    # 内容はリストだがJSONの文字列として渡されてくるのでstr型で受け取る。
    task_source: Annotated[
        str,
        Form(
            description="行動を説明するための作業内容を保持した文字列のリストです。",
            examples=['{"task_source": ["アップルパイにする。"]}'],
        ),
    ],
):
    try:
        source = json.loads(task_source)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="パラメータのJSONデコードに失敗しました。有効なJSON文字列を指定してください。",
        )

    prompt = f"""
            次の作業を完了するための行動を説明してください。
            # 作業内容
            {source}
            作業完了までに触れる必要のあるobjectの座標とobjectに対して必要なaction、
            及びobjectを特定するためのlabelを順番にリストにして返してください。
            以下のJSON形式に従ってください。
            # 出力形式
            [{{ "action": ["<action>", ...],
                "point": [y, x],
                "label": <label>}}, ...]
            座標は[y, x]形式で、0-1000に正規化してください。
            """

    results = {}

    for file in files:
        try:
            image_bytes = await file.read()
            response_json = _request_robotics_api(
                file=file,
                image_bytes=image_bytes,
                prompt=prompt,
            )

            _normalized_point_to_pixel_point(response_json, image_bytes)

            results[file.filename] = response_json
        except Exception as err:
            code = hasattr(err, "status_code") and getattr(err, "status_code")
            if not code:
                code = 500

            raise HTTPException(
                status_code=code,
                detail=f"画像処理エラー: ファイル {file.filename} の処理中にエラーが発生しました。{err=}, {type(err)=}",
            )

    return results
