from wallcrossing.core.models import Detection
from wallcrossing.runtime.roi import roi_from_polygon, translate_detection


def test_vertical_polygon_center_crops_half_width_full_height():
    polygon = [[450, 0], [550, 0], [550, 500], [450, 500]]

    roi = roi_from_polygon(polygon, (500, 1000), 0.5, 0.05)

    assert roi == (250, 0, 750, 500)


def test_vertical_polygon_near_left_clamps_to_frame_edge():
    polygon = [[50, 0], [100, 0], [100, 500], [50, 500]]

    roi = roi_from_polygon(polygon, (500, 1000), 0.5, 0.05)

    assert roi == (0, 0, 500, 500)


def test_vertical_polygon_near_right_clamps_to_frame_edge():
    polygon = [[900, 0], [950, 0], [950, 500], [900, 500]]

    roi = roi_from_polygon(polygon, (500, 1000), 0.5, 0.05)

    assert roi == (500, 0, 1000, 500)


def test_margin_can_make_roi_wider_than_half_frame():
    polygon = [[400, 0], [600, 0], [600, 500], [400, 500]]

    roi = roi_from_polygon(polygon, (500, 1000), 0.5, 0.2)

    assert roi == (200, 0, 800, 500)


def test_polygon_equal_to_minimum_is_not_expanded_by_margin():
    polygon = [[250, 0], [750, 0], [750, 500], [250, 500]]

    roi = roi_from_polygon(polygon, (500, 1000), 0.5, 0.2)

    assert roi == (250, 0, 750, 500)


def test_horizontal_polygon_auto_crops_half_height_full_width():
    polygon = [[0, 200], [1000, 200], [1000, 300], [0, 300]]

    roi = roi_from_polygon(polygon, (500, 1000), 0.5, 0.05)

    assert roi == (0, 125, 1000, 375)


def test_large_diagonal_polygon_auto_chooses_smaller_frame_fraction():
    polygon = [[0, 0], [260, 0], [800, 500], [0, 500]]

    roi = roi_from_polygon(polygon, (577, 982), 0.5, 0.0)

    assert roi == (0, 0, 800, 577)


def test_large_diagonal_polygon_is_not_expanded_when_already_over_minimum():
    polygon = [[0, 0], [260, 0], [800, 500], [0, 500]]

    roi = roi_from_polygon(polygon, (577, 982), 0.5, 0.05)

    assert roi == (0, 0, 800, 577)


def test_explicit_axis_overrides_auto_choice_for_diagonal_polygon():
    polygon = [[0, 0], [260, 0], [800, 500], [0, 500]]

    roi = roi_from_polygon(polygon, (577, 982), 0.5, 0.0, axis="y")

    assert roi == (0, 0, 982, 500)


def test_translate_detection_offsets_bbox_to_full_frame():
    det = Detection(bbox_xyxy=(10, 30, 50, 90), confidence=0.8, class_id=0)

    translated = translate_detection(det, 100, 20)

    assert translated.bbox_xyxy == (110, 50, 150, 110)
    assert translated.confidence == 0.8
    assert translated.class_id == 0
