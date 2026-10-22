"""Temporal aggregation: confidence thresholds and multi-frame confirmation of detections."""
import math

LOW_CONFIDENCE_CLASSES = ("cup", "bottle", "wine glass", "cell phone", "mouse", "remote")


def confidence_threshold(class_name):
    """Class-aware minimum confidence: low for small objects, stricter for people, 0.25 otherwise."""
    if class_name in LOW_CONFIDENCE_CLASSES:
        return 0.10
    if class_name == "person":
        return 0.30
    return 0.25


def group_detections(detections, max_distance=50):
    """Greedily group detections of the same class whose centres are within `max_distance` pixels
    of the group's first member. Each detection is a dict with 'class' and 'center'."""
    groups = []
    for det in detections:
        for group in groups:
            rep = group[0]
            if rep["class"] == det["class"] and math.dist(rep["center"], det["center"]) < max_distance:
                group.append(det)
                break
        else:
            groups.append([det])
    return groups


def frames_seen(group):
    """Number of distinct frames a group appears in (detections need a 'frame_idx')."""
    return len({d["frame_idx"] for d in group})


def confirm_groups(groups, min_frames=2):
    """Keep only groups seen in at least `min_frames` distinct frames."""
    return [g for g in groups if frames_seen(g) >= min_frames]
