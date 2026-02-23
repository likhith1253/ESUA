# ESUA System Features

ESUA offers a progressive series of computer vision and spatial logic features. It takes standard bounding boxes and converts them into semantic warnings.

## 1. Advanced Temporal Stability Filtering
**How it works:**  
The raw YOLO model inherently flickers when confidence thresholds dip between frames. ESUA maintains a rolling 5-frame buffer (`collections.deque`), requiring an object to persist across multiple frames before it's "confirmed" by the logic pipeline.

- **Inputs Supported:** Local multi-frame image buffer
- **Outputs Produced:** Clean, filtered list of stable entities with associated display names
- **Validation Logic:** Drops bounding boxes matching `count < CONFIRMATION_THRESHOLD_FRAMES`
- **Error Handling:** Fallback confidence mechanisms ensure the system retains the box state associated with the highest fidelity capture in the buffer.

## 2. Dynamic Class-Aware Thresholding
**How it works:**  
Most systems use a blanket confidence threshold (e.g., $0.5$). ESUA changes thresholds *per object category*.

- **Inputs Supported:** YOLO standard confidence score distribution
- **Outputs Produced:** Adjusted boolean acceptance for the bounding box
- **Validation Logic:** Small objects (`cup`, `bottle`, `cell phone`) are allowed to pass with a much lower confidence ($0.10$) to improve spatial recall, whereas larger ambiguous objects like `person` demand strict confidence ($0.30$) to prevent background hallucinations.

## 3. Pixel-Proximity Geometric Reasoning
**How it works:**  
Calculates accurate 2D centroid paths and maps the distance to a threshold value defining spatial states.

- **Inputs Supported:** Bounding box arrays `[x1,y1,x2,y2]`
- **Outputs Produced:** Linguistic relationship mapping (`"near"`, `"far from"`, `"overlaps with"`)
- **Validation Logic:** Euclidean spatial distance $ d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} $ compared against the $400$ pixel `NEAR_THRESHOLD`.

## 4. Semantic Risk Logic & Contextual Warnings
**How it works:**  
Translates raw nouns (like `cup` and `laptop`) into semantic traits (like `liquid` and `electronics`).

![Feature Example](images/feature_example.png)

- **Inputs Supported:** Proximity relationships overlapping with the `object_categories.py` dictionary mapping.
- **Outputs Produced:** High-priority visual and text alerts (e.g., `spill_risk`, `damage_risk`).
- **Validation Logic:** Logical boolean statements (`if liquid in categories and flammable in categories -> alert`).
- **Error Handling:** Defaults to harmless descriptions if an object map is missing or undefined in the rule bank.
