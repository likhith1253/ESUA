import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8n pre-trained model
print("Loading model...")
model = YOLO('yolov8n.pt')

# 2. Load the image
image_path = 'ESUA/phase1_object_detection/sample.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# 3. Run inference
print("Running inference...")
results = model(image)

# 4. Extract and Process Results
# results is a list, we only have one image so we take the first result
result = results[0]

# Iterate through detections
for box in result.boxes:
    # Extract Bounding Box Coordinates
    # box.xyxy provides [x1, y1, x2, y2]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Extract Class Name
    class_id = int(box.cls[0].item())
    class_name = result.names[class_id]
    
    # Extract Confidence Score
    confidence = box.conf[0].item()
    
    print(f"Detected: {class_name} | Confidence: {confidence:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")
    
    # 5. Draw Bounding Boxes and Labels
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Put text
    label = f"{class_name}: {confidence:.2f}"
    # Calculate text size to place the background box for text
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # Draw filled rectangle for text background for better visibility
    cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Save the result for verification
cv2.imwrite('ESUA/phase1_object_detection/result.jpg', image)
print("Result saved to ESUA/phase1_object_detection/result.jpg")

# Display the output
cv2.imshow('Object Detection', image)
print("Press any key to exit window...")
cv2.waitKey(0)
cv2.destroyAllWindows()



