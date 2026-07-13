from pipeline.frame_crop.cropper import _crop_inner_rect, _crop_outer_rect, select_cropper


def test_crop_inner_rect_returns_image_and_matching_rect(sample_image, sample_quad, make_config):
    result, rect = _crop_inner_rect(sample_image, sample_quad, make_config())
    assert rect == (100, 150, 600, 750)
    assert result.shape[:2] == (rect[3], rect[2])


def test_crop_outer_rect_returns_image_and_matching_rect(sample_image, sample_quad, make_config):
    result, rect = _crop_outer_rect(sample_image, sample_quad, make_config())
    assert result.shape[:2] == (rect[3], rect[2])
    x, y, w, h = rect
    assert x <= 100 and y <= 150
    assert x + w >= 700 and y + h >= 900


def test_margin_px_expands_rect_on_all_sides(sample_image, sample_quad, make_config):
    _, rect_no_margin = _crop_inner_rect(sample_image, sample_quad, make_config(margin_px=0))
    _, rect_margin = _crop_inner_rect(sample_image, sample_quad, make_config(margin_px=20))
    x0, y0, w0, h0 = rect_no_margin
    x1, y1, w1, h1 = rect_margin
    assert x1 == x0 - 20 and y1 == y0 - 20
    assert w1 == w0 + 40 and h1 == h0 + 40


def test_select_cropper_respects_inner_crop_flag(make_config):
    assert select_cropper(make_config(inner_crop=True)).name == "inner"
    assert select_cropper(make_config(inner_crop=False)).name == "outer"
