from wallcrossing.services.wall_contact import overlap_ratio, touches_wall


WALL = [[100, 300], [900, 300], [900, 500], [100, 500]]


def test_bbox_fully_outside_no_overlap():
    bbox = (200, 50, 260, 150)  # well above the wall band
    ratio = overlap_ratio(bbox, WALL, contact_mode="full_bbox")
    assert ratio == 0.0


def test_bbox_fully_inside_wall():
    bbox = (200, 320, 260, 480)  # entirely within wall rectangle
    ratio = overlap_ratio(bbox, WALL, contact_mode="full_bbox")
    assert ratio > 0.95


def test_bbox_partial_overlap():
    bbox = (200, 250, 260, 400)  # top half above wall, bottom half inside
    ratio = overlap_ratio(bbox, WALL, contact_mode="full_bbox")
    assert 0.3 < ratio < 0.7


def test_touches_wall_respects_threshold():
    bbox = (200, 250, 260, 400)
    hit_low, _ = touches_wall(bbox, WALL, min_overlap_ratio=0.1, contact_mode="full_bbox")
    hit_high, _ = touches_wall(bbox, WALL, min_overlap_ratio=0.99, contact_mode="full_bbox")
    assert hit_low is True
    assert hit_high is False


def test_bottom_band_only_uses_lower_part():
    # Person whose body is mostly above the wall but feet touch it.
    # bbox bottom at y=510 (just below wall bottom 500), top at y=100.
    bbox = (200, 100, 260, 510)
    full = overlap_ratio(bbox, WALL, contact_mode="full_bbox")
    band = overlap_ratio(bbox, WALL, contact_mode="bottom_band", bottom_band_ratio=0.25)
    # bottom band (y in [407.5, 510]) overlaps wall [300,500] more densely than full bbox
    assert band > full


def test_bottom_band_feet_above_wall_no_hit():
    # Feet end at y=290, above the wall top (300) -> bottom band shouldn't touch.
    bbox = (200, 100, 260, 290)
    hit, ratio = touches_wall(bbox, WALL, 0.02, contact_mode="bottom_band", bottom_band_ratio=0.25)
    assert hit is False
    assert ratio == 0.0
