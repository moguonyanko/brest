import json

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def normalize_bounding_box_with_image_pixel_size(
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
