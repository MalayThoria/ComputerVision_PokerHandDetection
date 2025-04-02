from ultralytics import YOLO
import cv2
import math
import cvzone

# Load the YOLO model. Adjust the path if necessary.
try:
    model = YOLO("../Yolo_weights/yolov8n.pt")
    print("YOLO model loaded successfully!")
except Exception as e:
    print("Error loading YOLO model:", e)
    exit()

# Read a test image
img = cv2.imread("img1.png")
if img is None:
    print("Test image not found. Please ensure 'test_image.jpg' exists in your directory.")
    exit()

# Run inference on the image
try:
    results = model(img, stream=True)
    print("YOLO inference completed!")
except Exception as e:
    print("Error during YOLO inference:", e)
    exit()

# Process and display the results
detected = False
for r in results:
    boxes = r.boxes
    for box in boxes:
        detected = True
        # Get coordinates, confidence, and class index
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])

        # Draw bounding box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cvzone.putTextRect(img, f'Class: {cls} {conf}', (max(0, x1), max(35, y1)))
        print(f"Detected object at ({x1}, {y1}, {x2}, {y2}) with confidence {conf} and class index {cls}")

if not detected:
    print("No detections found. The YOLO model might not be working properly on this image.")

# Display the image with detections
cv2.imshow("YOLO Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
