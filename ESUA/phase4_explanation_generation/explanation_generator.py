import cv2
import math
from ultralytics import YOLO
import object_categories
import explanation_templates

# 1. Load the YOLOv8n model
print("Loading model...")
model = YOLO('yolov8n.pt')

# 2. Load the image
image_path = 'ESUA/phase4_explanation_generation/sample.jpg'
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

# 4. Extract Objects and Categories
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    class_id = int(box.cls[0].item())
    class_name = result.names[class_id]
    
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    categories = object_categories.get_categories(class_name)
    
    objects.append({
        "name": class_name,
        "center": (cx, cy),
        "categories": categories
    })

print(f"Detected {len(objects)} objects. Analyzing context...\n")

# 5. Determine Risks and Generate Explanations
print("--- GENERATED EXPLANATIONS ---")

NEAR_THRESHOLD = 400
explanations_generated = False

for i in range(len(objects)):
    for j in range(i + 1, len(objects)):
        obj_a = objects[i]
        obj_b = objects[j]
        
        center_a = obj_a['center']
        center_b = obj_b['center']
        
        # Spatial Logic
        distance = math.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)
        is_near = distance < NEAR_THRESHOLD
        
        if is_near:
            risk_type = None
            
            cats_a = obj_a['categories']
            cats_b = obj_b['categories']
            
            # --- REDEFINING LOGIC TO RETURN RISK TYPES FOR TEMPLATES ---
            
            # Rule 1: Spill Risk (Liquid near Electronics)
            if ('liquid' in cats_a and 'electronics' in cats_b) or \
               ('liquid' in cats_b and 'electronics' in cats_a):
                risk_type = 'spill_risk'
                
            # Rule 2: Damage/Organization Risk (Liquid near Flammable)
            elif ('liquid' in cats_a and 'flammable' in cats_b) or \
                 ('liquid' in cats_b and 'flammable' in cats_a):
                risk_type = 'damage_risk'
                
            # We could add more rules here...
            
            if risk_type:
                # Prepare data for template
                # Ensure obj_a is the 'source' of risk for clearer phrasing if possible
                if 'liquid' in cats_b: 
                    # swap so obj_a is the liquid
                    t_obj_a, t_obj_b = obj_b, obj_a
                else:
                    t_obj_a, t_obj_b = obj_a, obj_b
                    
                context_data = {
                    'obj_a': t_obj_a['name'],
                    'cat_a': t_obj_a['categories'][0] if t_obj_a['categories'] else 'object',
                    'obj_b': t_obj_b['name'],
                    'cat_b': ','.join(t_obj_b['categories']) if t_obj_b['categories'] else 'uncategorized'
                }
                
                # Generate text
                explanation = explanation_templates.get_explanation(risk_type, context_data)
                print(explanation)
                print("-" * 40)
                explanations_generated = True

if not explanations_generated:
    print("✅ No specific risks requiring explanation were detected.")


