import cv2
import math
from ultralytics import YOLO
import object_categories
import risk_rules

# 1. Load the YOLOv8n model
print("Loading model...")
model = YOLO('yolov8n.pt')

# 2. Load the image
image_path = 'ESUA/phase3_context_reasoning/sample.jpg'
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

print("\n--- Detected Objects & Categories ---")
# 4. Extract Objects and Categories
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
    
    # Get Categories
    categories = object_categories.get_categories(class_name)
    
    # Store for later
    objects.append({
        "name": class_name,
        "box": (x1, y1, x2, y2),
        "center": (cx, cy),
        "categories": categories
    })
    
    cat_str = f"[{', '.join(categories)}]" if categories else "[Uncategorized]"
    print(f"Object: {class_name} | Center: ({cx}, {cy}) | Tags: {cat_str}")

# 5. Determine Relationships and Risks
print("\n--- Context & Risk Reasoning ---")

NEAR_THRESHOLD = 400

detected_risks = []

for i in range(len(objects)):
    for j in range(i + 1, len(objects)):
        obj_a = objects[i]
        obj_b = objects[j]
        
        center_a = obj_a['center']
        center_b = obj_b['center']
        
        # Calculate Euclidean Distance
        distance = math.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)
        
        # Determine Near/Far
        proximity = "near" if distance < NEAR_THRESHOLD else "far from"
        
        # CHECK RISKS
        # We pass the objects, distance, and the proximity string
        risks = risk_rules.check_risks(obj_a, obj_b, distance, proximity)
        
        if risks:
            for risk in risks:
                print(f"⚠️  {risk}")
                detected_risks.append(risk)

if not detected_risks:
    print("✅ No immediate risks detected.")


