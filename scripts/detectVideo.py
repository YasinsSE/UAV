import os
import cv2
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Predefined variables
confidence_score = 0.3  # Confidence threshold for detections
print(f"Confidence score: {confidence_score}")

color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_red = (0, 0, 255)
color_gray = (50, 50, 50)
color_yellow = (0, 255, 255)
color_green = (0, 255, 0)
target_box_color = color_red

font = cv2.FONT_HERSHEY_SIMPLEX
fps_font = cv2.FONT_HERSHEY_COMPLEX

total_fps = 0
average_fps = 0
num_of_frame = 0

text_AH = "AH: Lock Rectangle"
text_AV = "AV: Strike Zone"
text_AK = "AK: Camera FoV"

class_name = 'UAV'

#video_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/data/inference/videos/test_3840_2160_30fps.mp4"
video_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/data/inference/videos/testervideo.mov"
model_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/models/trained/best_yolo10m.pt"
#model_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/models/trained/best_m50k_trained.pt"
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
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Width: {width}, Height: {height}, Frames: {total_frames}, FPS: {fps}")

yellow_rect_x1 = int(0.25 * width)
yellow_rect_y1 = int(0.10 * height)
yellow_rect_x2 = int(0.75 * width)
yellow_rect_y2 = int(0.90 * height)
print(f"Yellow rectangle; x1:{yellow_rect_x1}, y1:{yellow_rect_y1}, x2:{yellow_rect_x2}, y2:{yellow_rect_y2}")

# Initialize VideoWriter to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

model = YOLO(model_path)

"""
# Create a binary mask for locking
mask = cv2.imread("mask.png", 0)  # Load a pre-defined mask image

masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
results = model(masked_frame, verbose=False)[0]
"""

# Running video
print("Processing started at", datetime.now())
while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Warning: Skipped a frame.")
        break

    results = model(frame, verbose=False, stream=True)
    detected = False

    # Process each result
    for result in results:
        boxes = np.array(result.boxes.data.tolist())  # Get bounding boxes
        if boxes.size > 0:
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

                if result.names[class_id] == class_name and score > confidence_score:

                    detected = True
                    log_message = f"Detected {class_name}: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, score: {score:.2f})\n"
                    log.write(log_message)
                    log.flush()  # Ensure immediate save

                    cv2.rectangle(frame, (x1, y1), (x2, y2), target_box_color, 3)

                    score_text = f"{class_name}: {score * 100:.1f}%"
                    label_size, baseLine = cv2.getTextSize(score_text, font, 1, 2)
                    background_top_left = (x1, y1 - label_size[1] - 5)
                    background_bottom_right = (x1 + label_size[0] + 5, y1 + baseLine - 20)

                    cv2.rectangle(frame, background_top_left, background_bottom_right, color_gray, cv2.FILLED)

                    text_loc = (x1 + 2, y1 - 3)
                    cv2.putText(frame, score_text, text_loc, font, 2, color_white, thickness=3)

                    text_loc_AH = (x1, y2 + 50)
                    cv2.putText(frame, text_AH, text_loc_AH, font, 1.5, color_black, thickness=3)

    if not detected:
        print("No objects detected in this frame.")

    # text_loc_AV = (yellow_rect_x1+25, yellow_rect_y2 - 25)
    text_loc_AV = (yellow_rect_x1 + int(width / 120), yellow_rect_y2 - int(height / 60))
    cv2.putText(frame, text_AV, text_loc_AV, font, 2.2, color_black, thickness=3)

    text_loc_AK = (30, height - 40)
    cv2.putText(frame, text_AK, text_loc_AK, font, 2.2, color_black, thickness=3)

    # Calculate and display FPS
    end = time.time()
    fps_calc = 1 / (end - start)
    total_fps += fps_calc
    num_of_frame += 1
    average_fps = total_fps / num_of_frame
    avg_fps = float(f"{average_fps:.2f}")

    cv2.putText(frame, "Processed FPS: " + str(avg_fps), (50, 100), fps_font, 2, color_red, thickness=2)
    cv2.putText(frame, "Actual FPS: " + str(int(fps)), (50, 180), fps_font, 2, color_red, thickness=2)

    # Write the frame to the output video
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
