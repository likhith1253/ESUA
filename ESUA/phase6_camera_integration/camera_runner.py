import cv2
import time
import math
from ultralytics import YOLO
import object_categories
import risk_rules
import explanation_templates

def main():
    print("Initializing ESUA Camera Runner...")
    print("Press 'q' to quit.")

    # 1. Load Model
    # Using YOLOv8n for speed on CPU
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open Camera
    # Index 0 is usually the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        print("Please check if your camera is connected and not used by another app.")
        print("Exiting...")
        return

    print("✅ Camera opened successfully.")

    # Performance settings
    frame_count = 0
    SKIP_FRAMES = 5  # Run inference every 5 frames to keep UI responsive
    
    # Store last known risks to display during skipped frames
    current_explanations = []
    current_boxes = [] # Store boxes: (x1, y1, x2, y2, label, color)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Resize for performance (optional, but good for CPU)
        # width=640 is standard specific for YOLOv8
        frame = cv2.resize(frame, (640, 480))
        
        # Increment frame counter
        frame_count += 1
        
        # --- ML PIPELINE (Run only every N frames) ---
        if frame_count % SKIP_FRAMES == 0:
            current_explanations = []
            current_boxes = []
            
            # A. Detection
            results = model(frame, verbose=False) # verbose=False to reduce console spam
            result = results[0]
            
            objects = []
            
            # Extract data
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                categories = object_categories.get_categories(class_name)
                
                obj_data = {
                    "name": class_name,
                    "center": (cx, cy),
                    "categories": categories,
                    "box": (x1, y1, x2, y2)
                }
                objects.append(obj_data)
                
                # Add to display list
                # Color based on risk status could be added, for now Green
                current_boxes.append((x1, y1, x2, y2, class_name, (0, 255, 0)))

            # B. Spatial & Risk Reasoning
            NEAR_THRESHOLD = 300 # Pixels (adjusted for webcam resolution)
            
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    obj_a = objects[i]
                    obj_b = objects[j]
                    
                    center_a = obj_a['center']
                    center_b = obj_b['center']
                    
                    distance = math.sqrt((center_a[0] - center_b[0])**2 + (center_a[1] - center_b[1])**2)
                    
                    if distance < NEAR_THRESHOLD:
                         # Check Risks
                        risk_type = None
                        cats_a = obj_a['categories']
                        cats_b = obj_b['categories']
                        
                        if ('liquid' in cats_a and 'electronics' in cats_b) or \
                           ('liquid' in cats_b and 'electronics' in cats_a):
                            risk_type = 'spill_risk'
                        elif ('liquid' in cats_a and 'flammable' in cats_b) or \
                             ('liquid' in cats_b and 'flammable' in cats_a):
                            risk_type = 'damage_risk'
                            
                        # If risk detected, generate explanation
                        if risk_type:
                            # Context swap for template
                            if 'liquid' in cats_b: 
                                t_obj_a, t_obj_b = obj_b, obj_a
                            else:
                                t_obj_a, t_obj_b = obj_a, obj_b
                                
                            context_data = {
                                'obj_a': t_obj_a['name'],
                                'cat_a': t_obj_a['categories'][0] if t_obj_a['categories'] else 'object',
                                'obj_b': t_obj_b['name'],
                                'cat_b': ','.join(t_obj_b['categories'])
                            }
                            
                            # Get full text
                            full_expl = explanation_templates.get_explanation(risk_type, context_data)
                            
                            # Just take the first line (Observation) and last (Suggestion) for on-screen display to save space
                            lines = full_expl.split('\n')
                            short_text = f"⚠️ {lines[0]} -> {lines[-1]}"
                            current_explanations.append(short_text)
                            
                            # Change box color to Red for involved objects
                            # (Simplified: we won't update the box list for now to keep code simple)

        # --- DISPLAY LOOP (Runs every frame) ---
        
        # 1. Draw Boxes
        for (x1, y1, x2, y2, label, color) in current_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # 2. Draw Explanations (Overlay)
        if current_explanations:
            # Draw a background pane for text
            start_y = 30
            for i, text in enumerate(current_explanations):
                # Only show top 3 to avoid clutter
                if i >= 3: break
                
                # Simple text drawing
                cv2.putText(frame, text, (10, start_y + (i * 25)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                            
        # Show Frame
        cv2.imshow('ESUA Real-Time Assistant', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera runner stopped.")

if __name__ == "__main__":
    main()
