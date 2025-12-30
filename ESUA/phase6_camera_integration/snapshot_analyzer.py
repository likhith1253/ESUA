
import cv2
import math
import sys
import os
import collections
import numpy as np
from ultralytics import YOLO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import helper modules
try:
    import object_categories
    import risk_rules
    import explanation_templates
except ImportError:
    # Fallback to importing from previous phases
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../phase4_explanation_generation'))
        import object_categories
        import risk_rules
        import explanation_templates
    except ImportError:
        print("Error: Could not import ESUA helper modules.")
        sys.exit(1)

# --- CONFIGURATION ---
BUFFER_SIZE = 5
CONFIRMATION_THRESHOLD_FRAMES = 2  # Object must be seen in at least this many frames
GROUPING_DISTANCE_THRESHOLD = 50   # Pixels

# Class-Aware Thresholds
def get_confidence_threshold(class_name):
    # Lower threshold for small/hard objects
    if class_name in ['cup', 'bottle', 'wine glass', 'cell phone', 'mouse', 'remote']:
        return 0.10
    # Stricter for people to avoid ghosts
    elif class_name == 'person':
        return 0.30
    # Standard for others
    return 0.25

def main():
    print("Initializing Robust ESUA Camera System...")
    print("Controls:\n  'c' - Capture (Multi-Frame Analysis)\n  'q' - Quit")
    
    # 1. CAMERA SETUP
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Ring Buffer
    frame_buffer = collections.deque(maxlen=BUFFER_SIZE)
    
    capture_triggered = False
    final_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add to buffer
        frame_buffer.append(frame)

        # Display
        cv2.imshow('ESUA Live Feed (Buffering 5 Frames)', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if len(frame_buffer) < BUFFER_SIZE:
                print("Buffer filling... wait a moment.")
                continue
            capture_triggered = True
            print("Capturing burst of frames for analysis...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if not capture_triggered:
        return

    # 2. MULTI-FRAME ANALYSIS
    print("\n" + "="*50)
    print("� ROBUSTNESS PHASE: MULTI-FRAME AGGREGATION")
    print("="*50)
    
    model = YOLO('yolov8n.pt')
    
    # Store all detections from all frames
    # Structure: { 'frame_idx': int, 'class': str, 'box': tuple, 'conf': float, 'center': tuple }
    all_detections = []
    
    # Use the last frame as the "Reference Frame" for display
    reference_frame_idx = len(frame_buffer) - 1
    reference_image = frame_buffer[reference_frame_idx].copy()
    
    print(f"Analyzing {len(frame_buffer)} frames...")
    
    for f_idx, frame in enumerate(frame_buffer):
        results = model(frame, verbose=False) # valid=False to reduce spam
        result = results[0]
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = box.conf[0].item()
            
            # --- DEBUG LOGGING (Before Threshold) ---
            # print(f"DEBUG: Frame {f_idx} Raw: {cls_name} ({conf:.2f})")
            
            # --- CLASS-AWARE THRESHOLDING ---
            thresh = get_confidence_threshold(cls_name)
            if conf >= thresh:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                all_detections.append({
                    'frame_idx': f_idx,
                    'class': cls_name,
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'center': (cx, cy)
                })
                # print(f"  -> ACCEPTED (Thresh {thresh})")
            else:
                pass
                # print(f"  -> REJECTED (Thresh {thresh})")

    # 3. AGGREGATION LOGIC
    # Group detections that are spatially close and same class
    print(f"\nAggregating {len(all_detections)} candidates across temporal buffer...")
    
    candidate_groups = [] # List of lists of detection dicts
    
    for det in all_detections:
        matched = False
        for group in candidate_groups:
            # Check representative (first item in group)
            rep = group[0]
            
            if rep['class'] == det['class']:
                # Check spatial distance
                dist = math.sqrt((rep['center'][0] - det['center'][0])**2 + 
                                 (rep['center'][1] - det['center'][1])**2)
                
                if dist < GROUPING_DISTANCE_THRESHOLD:
                    group.append(det)
                    matched = True
                    break
        
        if not matched:
            candidate_groups.append([det])
            
    # Filter Groups by temporal consistency
    confirmed_objects = []
    
    print("\n--- Objects Confirmation Status ---")
    for group in candidate_groups:
        # Count unique frames
        frames_seen = set(d['frame_idx'] for d in group)
        count = len(frames_seen)
        
        rep = group[0] # Representative for naming
        cls_name = rep['class']
        
        status = "CONFIRMED" if count >= CONFIRMATION_THRESHOLD_FRAMES else "DISCARDED (Transient/Noise)"
        print(f"Object '{cls_name}': Seen in {count}/{BUFFER_SIZE} frames -> {status}")
        
        if count >= CONFIRMATION_THRESHOLD_FRAMES:
            # Select the detection from the Reference Frame (most recent) if available,
            # otherwise the one with highest confidence
            
            # Try to find detection in reference frame
            best_det = None
            for d in group:
                if d['frame_idx'] == reference_frame_idx:
                    best_det = d
                    break
            
            # Fallback: Highest confidence
            if not best_det:
                best_det = max(group, key=lambda x: x['conf'])
                
            # Normalize Name Logic (Optional Step 4 from requirements)
            display_name = best_det['class']
            if display_name in ['cup', 'bottle', 'glass']:
                display_name = 'liquid container' # Example normalization
            
            # Add to final list
            confirmed_objects.append({
                "name": best_det['class'], # Original class for risk logic lookup
                "display_name": display_name,
                "box": best_det['box'],
                "center": best_det['center'],
                "conf": best_det['conf'],
                "frames_count": count
            })

    # 4. RUN ESUA PIPELINE ON CONFIRMED OBJECTS
    print("\n" + "="*50)
    print("🚀 RUNNING ESUA PIPELINE (Spatial & Risk)")
    print("="*50)
    
    # Prepare objects for Spatial/Risk logic
    # Need to add categories
    processed_objects = []
    for obj in confirmed_objects:
        categories = object_categories.get_categories(obj['name'])
        
        # Draw on Reference Image
        x1, y1, x2, y2 = obj['box']
        cv2.rectangle(reference_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label: Name + Conf + Stability
        label = f"{obj['display_name']} ({obj['conf']:.2f}) [{obj['frames_count']}f]"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(reference_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(reference_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        processed_objects.append({
            "name": obj['name'], # Use original for consistency with rules
            "display_name": obj['display_name'],
            "box": obj['box'],
            "center": obj['center'],
            "categories": categories,
            "conf": obj['conf']
        })
        
        print(f"• {obj['display_name']} (Stability: {obj['frames_count']}/5 frames)")

    # Spatial Logic (Phase 2)
    print("\n[Phase 2] Spatial Relationships:")
    NEAR_THRESHOLD = 400
    relationships = []
    
    for i in range(len(processed_objects)):
        for j in range(i + 1, len(processed_objects)):
            obj_a = processed_objects[i]
            obj_b = processed_objects[j]
            
            dist = math.sqrt((obj_a['center'][0] - obj_b['center'][0])**2 + 
                             (obj_a['center'][1] - obj_b['center'][1])**2)
            
            proximity = "near" if dist < NEAR_THRESHOLD else "far from"
            
            print(f"- {obj_a['display_name']} is {proximity} {obj_b['display_name']} ({dist:.1f}px)")
            
            relationships.append({
                "obj_a": obj_a,
                "obj_b": obj_b,
                "distance": dist,
                "proximity": proximity
            })
            
    # Risk & Explanation Logic (Phase 3 & 4)
    print("\n[Phase 3 & 4] Risk Analysis:")
    risks_found = False
    
    for rel in relationships:
        obj_a = rel['obj_a']
        obj_b = rel['obj_b']
        proximity = rel['proximity']
        
        risk_type = None
        cats_a = obj_a['categories']
        cats_b = obj_b['categories']
        
        if proximity == "near":
            if ('liquid' in cats_a and 'electronics' in cats_b) or \
               ('liquid' in cats_b and 'electronics' in cats_a):
                risk_type = 'spill_risk'
            elif ('liquid' in cats_a and 'flammable' in cats_b) or \
                 ('liquid' in cats_b and 'flammable' in cats_a):
                risk_type = 'damage_risk'
        
        if risk_type:
            risks_found = True
            
            # Template Prep
            if 'liquid' in cats_b:
                t_obj_a, t_obj_b = obj_b, obj_a
            else:
                t_obj_a, t_obj_b = obj_a, obj_b
                
            context_data = {
                'obj_a': t_obj_a['display_name'],
                'cat_a': t_obj_a['categories'][0] if t_obj_a['categories'] else 'object',
                'obj_b': t_obj_b['display_name'],
                'cat_b': ','.join(t_obj_b['categories'])
            }
            
            explanation = explanation_templates.get_explanation(risk_type, context_data)
            print(f"⚠️  {risk_type.replace('_', ' ').upper()}: {explanation}")

    if not risks_found:
        print("✅ No immediate risks detected.")

    # Save and Show
    output_path = 'ESUA/phase6_camera_integration/result_robust.jpg'
    cv2.imwrite(output_path, reference_image)
    print(f"\nSaved robust analysis result to {output_path}")
    
    cv2.imshow('ESUA Robust Analysis', reference_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
