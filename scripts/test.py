import os
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Predefined variables
confidence_score = 0.2  # Confidence threshold for detections
print(f"Confidence score: {confidence_score}")

# Colors for visualization
color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_red = (0, 0, 255)
color_green = (0, 255, 0)

# Fonts
font = cv2.FONT_HERSHEY_SIMPLEX
fps_font = cv2.FONT_HERSHEY_COMPLEX

# Paths for video and model
video_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/data/inference/videos/testervideo.mov"
model_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/models/trained/best_yolo10m.pt"
save_dir = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/runs/detect/results/videos/"
os.makedirs(save_dir, exist_ok=True)

# Generate a unique filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{timestamp}_processed_uav_detection.mp4"
save_path = os.path.join(save_dir, filename)

log_dir = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/runs/detect/logs/"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{timestamp}_detection_log.txt")
log = open(log_file, "w")

# Load video
cap = cv2.VideoCapture(video_path)

# Check if the video is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Width: {width}, Height: {height}, FPS: {fps}")

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

# Load YOLO model
model = YOLO(model_path)

# Initialize DeepSORT tracker
# Initialize DeepSORT tracker
deepsort = DeepSort(
    embedder="mobilenet",  # Use a valid embedder like 'mobilenet' or 'clip'
    max_age=30,
    n_init=3,
    nn_budget=100,
    max_iou_distance=0.7
)


# Running video
print("Processing started at", datetime.now())
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Warning: Skipped a frame.")
        break

    # YOLO detection
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2, score, class_id = box.xyxy.tolist() + [box.conf.item(), box.cls.item()]
                x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

                if result.names[class_id] == class_name and score > confidence_score:
                    detections.append([x1, y1, x2, y2, score])
                    # Log detection
                    log_message = f"Detected {class_name}: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, score: {score:.2f})\n"
                    log.write(log_message)
                    log.flush()

    # Prepare detections for DeepSORT
    bbox_xywh = []
    confidences = []
    for det in detections:
        x1, y1, x2, y2, conf = det
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])  # Convert to [cx, cy, w, h]
        confidences.append(conf)

    # Update tracker
    outputs = deepsort.update_tracks(np.array(bbox_xywh), np.array(confidences), frame)

    # Draw tracked objects
    for track in outputs:
        if track.is_confirmed() and track.time_since_update == 0:
            bbox = track.to_tlbr()  # Get [x1, y1, x2, y2]
            obj_id = track.track_id
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_green, 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), font, 0.8, color_green, 2)

    # Calculate and display FPS
    end = time.time()
    fps_calc = 1 / (end - start)
    total_fps += fps_calc
    num_of_frame += 1
    avg_fps = total_fps / num_of_frame

    cv2.putText(frame, f"Processed FPS: {avg_fps:.2f}", (50, 100), fps_font, 1, color_red, 2)

    # Save frame to output video
    out.write(frame)
    cv2.imshow("Processed Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
log.close()

print(f"Logs saved to: {log_file}")
print(f"Processing finished at {datetime.now()}")
print(f"Video saved to: {save_path}")
