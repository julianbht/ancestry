import numpy as np

from pipeline.frame_crop.geometry import inner_bounding_rect, order_points


def test_order_points_returns_tl_tr_br_bl():
    # Corners given out of order; order_points must sort them tl, tr, br, bl.
    quad = np.array([[700, 900], [100, 150], [700, 150], [100, 900]], dtype=np.float32)
    tl, tr, br, bl = order_points(quad)
    assert tuple(tl) == (100, 150)
    assert tuple(tr) == (700, 150)
    assert tuple(br) == (700, 900)
    assert tuple(bl) == (100, 900)


def test_inner_bounding_rect_axis_aligned_quad():
    quad = np.array([[100, 150], [700, 150], [700, 900], [100, 900]], dtype=np.float32)
    x, y, w, h = inner_bounding_rect(quad)
    assert (x, y, w, h) == (100, 150, 600, 750)


def test_inner_bounding_rect_clips_corners_of_rotated_quad():
    # A slightly rotated quad: the inner rect must stay fully inside it, so it
    # clips a few pixels off each side rather than including any background.
    quad = np.array(
        [[110, 150], [700, 160], [690, 900], [100, 890]], dtype=np.float32
    )
    x, y, w, h = inner_bounding_rect(quad)
    assert x >= 110 and y >= 160
    assert x + w <= 690 and y + h <= 890
