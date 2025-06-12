import time
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import requests

# Function to update server value
def update_server(value):
    try:
        print(f"Sending value {value} to server...")
        response = requests.post("http://localhost:5000/update", json={"value": value})
        print(f"Server response: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Server update failed: {e}")

# Load the model
model = YOLO("src/yolo11n.onnx", task="detect")

# Load configuration
with open("src/properties.yaml", "r") as properties_file:
    properties = yaml.safe_load(properties_file)

# Track person presence time
person_present = False
person_start_time = None
person_reported_stayed = False

# Function to detect and draw only the nearest person
def detect_nearest_person(frame):
    results = model(frame)
    nearest_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name.lower() == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    nearest_box = box

    if nearest_box:
        x1, y1, x2, y2 = map(int, nearest_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if properties["SHOW_CLASS_NAME"]:
            label = f"person"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return nearest_box, frame

# Start camera
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    nearest_person_box, annotated_frame = detect_nearest_person(frame)
    current_time = time.time()

    if nearest_person_box is not None:
        if not person_present:
            person_present = True
            person_start_time = current_time
            person_reported_stayed = False
        elif not person_reported_stayed and (current_time - person_start_time >= 2.5):
            print("Person stayed")
            update_server(1)
            person_reported_stayed = True
    else:
        if person_present:
            print("Person left")
            update_server(0)
            person_present = False
            person_start_time = None
            person_reported_stayed = False

    cv2.imshow("Nearest Person Detection", annotated_frame)
    time.sleep(0.03)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
