"""
設定ファイルからモデル名を取得する関数群
"""

from utils import load_json

# Gemini APIのモデル設定読み込み
genaiapi_models = load_json(path="genaiapi_models.json")
genaiapi_model_names = genaiapi_models["model_name"]


def get_generate_text_model_name() -> str:
    return genaiapi_model_names["generate_text"]


def get_generate_image_model_name() -> str:
    return genaiapi_model_names["generate_image"]


def get_generate_vision_model_name() -> str:
    return genaiapi_model_names["vision"]


def get_generate_transcription_movie_model_name() -> str:
    return genaiapi_model_names["transcription_movie"]


def get_generate_transcription_movie_inline_model_name() -> str:
    return genaiapi_model_names["transcription_movie_inline"]


def get_generate_transcription_audio_inline_model_name() -> str:
    return genaiapi_model_names["transcription_audio_inline"]


def get_model_name_summarize_document() -> str:
    return genaiapi_model_names["summarize_document"]


def get_model_name_text_embedding() -> str:
    return genaiapi_model_names["text_embedding"]


def get_model_name_thinking() -> str:
    return genaiapi_model_names["thinking"]


def get_model_generate_speech() -> str:
    return genaiapi_model_names["generate_speech"]


def get_model_url_context() -> str:
    return genaiapi_model_names["url_context"]


def get_model_live_api_speech() -> str:
    return genaiapi_model_names["live_api_speech"]


def get_model_robotics() -> str:
    return genaiapi_model_names["robotics"]
