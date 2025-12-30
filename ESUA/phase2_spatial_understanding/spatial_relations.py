import cv2
import math
from ultralytics import YOLO

# 1. Load the YOLOv8n model
print("Loading model...")
model = YOLO('yolov8n.pt')

# 2. Load the image
image_path = 'ESUA/phase2_spatial_understanding/sample.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# 3. Run inference
print("Running inference...")
results = model(image)
result = results[0]

# List to store detected object details
objects = []

print("\n--- Detected Objects ---")
# 4. Extract Objects and Calculate Centers
for box in result.boxes:
    # Coordinates
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Class Name
    class_id = int(box.cls[0].item())
    class_name = result.names[class_id]
    
    # Calculate Center Point
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Store for later
    objects.append({
        "name": class_name,
        "box": (x1, y1, x2, y2),
        "center": (cx, cy)
    })
    
    print(f"Object: {class_name} | Center: ({cx}, {cy})")

# 5. Determine Spatial Relationships
print("\n--- Spatial Relationships ---")

# Threshold for "Near" (pixels) - this is a simple heuristic
NEAR_THRESHOLD = 400 

for i in range(len(objects)):
    for j in range(i + 1, len(objects)):
        obj_a = objects[i]
        obj_b = objects[j]
        
        name_a = obj_a['name']
        name_b = obj_b['name']
        
        center_a = obj_a['center']
        center_b = obj_b['center']
        
        # Calculate Euclidean Distance
        distance = math.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)
        
        # Determine Near/Far
        proximity = "near" if distance < NEAR_THRESHOLD else "far from"
        
        # Determine Left/Right (based on X coordinate)
        if center_a[0] < center_b[0]:
            horizontal_rel = f"{name_a} is to the left of {name_b}"
        else:
            horizontal_rel = f"{name_a} is to the right of {name_b}"
            
        # Determine Overlapping
        # Two rectangles do NOT overlap if one is to the right of the other, 
        # or one is above the other.
        box_a = obj_a['box'] # x1, y1, x2, y2
        box_b = obj_b['box']
        
        # Check for NO overlap conditions
        no_overlap = (box_a[2] < box_b[0] or  # A right < B left
                      box_a[0] > box_b[2] or  # A left > B right
                      box_a[3] < box_b[1] or  # A bottom < B top
                      box_a[1] > box_b[3])    # A top > B bottom
        
        overlap_status = "overlaps with" if not no_overlap else "does not overlap"
        
        # Print the sentence
        print(f"- {horizontal_rel}")
        print(f"- {name_a} is {proximity} {name_b} (Distance: {distance:.2f})")
        if not no_overlap:
             print(f"- {name_a} {overlap_status} {name_b}")
        print("---")


