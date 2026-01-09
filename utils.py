import json
import pathlib

def load_json(file_path: str):
    path = pathlib.Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path.absolute()}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSONの形式が正しくありません（{e.lineno}行目付近）: {e.msg}")

def convert_normalized_bbox_to_pixel_bbox(
    bounding_box: dict,
    original_image_size: tuple[int, int],
    scale: int,
) -> dict:
    """
    正規化されたバウンディングボックスを画像のピクセルサイズに基づいて変換します。
    Args:
        bounding_box (dict): 正規化されたバウンディングボックス。形式は
                             {"bounding_box": [ymin, xmin, ymax, xmax]}。
        size (tuple[int, int]): 画像の元のサイズ (width, height)。
        scale (int): 正規化に使用されたスケール値。
    Returns:
        dict: ピクセル単位に変換されたバウンディングボックス。形式は
              {"bounding_box": [ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel]}。
    """
    ymin_norm, xmin_norm, ymax_norm, xmax_norm = bounding_box
    original_width, original_height = original_image_size

    xmin_pixel = int(round((xmin_norm / scale) * original_width))
    ymin_pixel = int(round((ymin_norm / scale) * original_height))
    xmax_pixel = int(round((xmax_norm / scale) * original_width))
    ymax_pixel = int(round((ymax_norm / scale) * original_height))

    return [ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel]


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def convert_normalized_point_to_pixel_point(
    normalized_point: Point, original_image_size: tuple[int, int], scale: int
) -> Point:
    """
    正規化されたポイントを画像のピクセルサイズに基づいて変換します。
    Args:
        point (Point): 正規化されたポイント。
        original_image_size (tuple[int, int]): 画像の元のサイズ (width, height)。
        scale (int): 正規化に使用されたスケール値。
    Returns:
        Point: ピクセル単位に変換されたポイント。
    """
    original_width, original_height = original_image_size
    x_pixel = int((normalized_point.x / scale) * original_width)
    y_pixel = int((normalized_point.y / scale) * original_height)

    return Point(x_pixel, y_pixel)
