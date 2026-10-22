"""Geometry helpers shared by the ESUA phases. Boxes are (x1, y1, x2, y2) in pixels."""
import math

NEAR_THRESHOLD = 400  # pixels between box centres; a simple 2D heuristic, not a real-world distance


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(center_a, center_b):
    return math.sqrt((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2)


def proximity(dist, threshold=NEAR_THRESHOLD):
    """'near' when the centres are closer than `threshold` pixels, otherwise 'far from'."""
    return "near" if dist < threshold else "far from"


def boxes_overlap(box_a, box_b):
    """True when two boxes intersect (touching edges count as overlapping, as in the phase 2 script)."""
    return not (box_a[2] < box_b[0] or box_a[0] > box_b[2] or box_a[3] < box_b[1] or box_a[1] > box_b[3])


def horizontal_relation(center_a, center_b):
    """'left of' if A's centre is left of B's, else 'right of'."""
    return "left of" if center_a[0] < center_b[0] else "right of"
