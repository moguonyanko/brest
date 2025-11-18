from utils import convert_normalized_bbox_to_pixel_bbox

def test_normalize_bounding_box_with_image_pixel_size():

    bounding_box = [431, 220, 512, 296]  # [ymin, xmin, ymax, xmax] normalized
    original_image_size = (800, 600)  # width, height
    scale = 1000

    result = convert_normalized_bbox_to_pixel_bbox(
        bounding_box,
        original_image_size,
        scale,
    )

    # [ymin_pixel, xmin_pixel, ymax_pixel, xmax_pixel]
    assert result == [259, 176, 307, 237]  

