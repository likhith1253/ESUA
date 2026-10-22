"""Unit tests for the pure-Python ESUA logic (no webcam, OpenCV or model weights needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from esua_core import aggregation, categories, explanations, risk_rules, spatial


def obj(name, box):
    return {"name": name, "box": box, "center": spatial.center(box), "categories": categories.get_categories(name)}


# ---- categories
def test_categories():
    assert categories.get_categories("cup") == ["liquid"]
    assert categories.get_categories("laptop") == ["electronics"]
    assert categories.get_categories("book") == ["flammable"]
    assert categories.get_categories("giraffe") == []


# ---- spatial
def test_center_and_distance():
    assert spatial.center((0, 0, 10, 20)) == (5, 10)
    assert spatial.distance((0, 0), (3, 4)) == 5.0


def test_proximity_threshold_is_strict():
    assert spatial.proximity(399.9) == "near"
    assert spatial.proximity(400) == "far from"
    assert spatial.proximity(10, threshold=5) == "far from"


def test_boxes_overlap():
    assert spatial.boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15))
    assert spatial.boxes_overlap((0, 0, 10, 10), (10, 10, 20, 20))  # touching counts
    assert not spatial.boxes_overlap((0, 0, 10, 10), (11, 0, 20, 10))
    assert not spatial.boxes_overlap((0, 0, 10, 10), (0, 11, 10, 20))


def test_horizontal_relation():
    assert spatial.horizontal_relation((1, 0), (5, 0)) == "left of"
    assert spatial.horizontal_relation((9, 0), (5, 0)) == "right of"


# ---- risk rules
def test_spill_risk_when_liquid_near_electronics_either_order():
    cup, laptop = obj("cup", (0, 0, 50, 50)), obj("laptop", (100, 0, 300, 150))
    d = spatial.distance(cup["center"], laptop["center"])
    rel = spatial.proximity(d)
    assert rel == "near"
    assert risk_rules.classify_risk(cup["categories"], laptop["categories"], rel) == "spill_risk"
    assert risk_rules.classify_risk(laptop["categories"], cup["categories"], rel) == "spill_risk"
    msgs = risk_rules.check_risks(cup, laptop, d, rel)
    assert len(msgs) == 1 and msgs[0].startswith("Spill Risk")


def test_damage_risk_when_liquid_near_flammable():
    assert risk_rules.classify_risk(["liquid"], ["flammable"], "near") == "damage_risk"
    msgs = risk_rules.check_risks(obj("bottle", (0, 0, 10, 10)), obj("book", (20, 0, 40, 10)), 25, "near")
    assert len(msgs) == 1 and msgs[0].startswith("Damage Risk")


def test_no_risk_when_far_or_unrelated():
    assert risk_rules.classify_risk(["liquid"], ["electronics"], "far from") is None
    assert risk_rules.classify_risk(["electronics"], ["flammable"], "near") is None
    assert risk_rules.classify_risk([], [], "near") is None
    assert risk_rules.check_risks(obj("cup", (0, 0, 1, 1)), obj("laptop", (0, 0, 1, 1)), 900, "far from") == []


# ---- explanations
def test_explanation_fills_template_and_falls_back():
    text = explanations.get_explanation("spill_risk", {"obj_a": "cup", "obj_b": "laptop"})
    assert "A cup is placed close to a laptop." in text and len(text.splitlines()) == 4
    assert explanations.get_explanation("unknown_type", {"obj_a": "x", "obj_b": "y"}).startswith("A x is near a y.")
    assert "[Missing data" in explanations.get_explanation("spill_risk", {"obj_a": "cup"})


# ---- temporal aggregation
def test_confidence_thresholds():
    assert aggregation.confidence_threshold("cup") == 0.10
    assert aggregation.confidence_threshold("person") == 0.30
    assert aggregation.confidence_threshold("laptop") == 0.25


def det(frame, cls, center):
    return {"frame_idx": frame, "class": cls, "center": center}


def test_grouping_and_confirmation():
    dets = [det(0, "cup", (100, 100)), det(1, "cup", (110, 104)),  # same cup, two frames
            det(2, "cup", (400, 400)),                              # a different cup seen once
            det(0, "laptop", (100, 100))]                           # same place, different class
    groups = aggregation.group_detections(dets, max_distance=50)
    assert len(groups) == 3
    confirmed = aggregation.confirm_groups(groups, min_frames=2)
    assert len(confirmed) == 1 and confirmed[0][0]["class"] == "cup"


def test_same_frame_duplicates_do_not_confirm():
    groups = aggregation.group_detections([det(0, "cup", (0, 0)), det(0, "cup", (5, 5))])
    assert len(groups) == 1 and aggregation.frames_seen(groups[0]) == 1
    assert aggregation.confirm_groups(groups, min_frames=2) == []
