import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Predefined variables
confidence_score = 0.6  # Confidence threshold for detections

color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_red = (0, 0, 255)
color_gray = (50, 50, 50)
color_yellow = (0, 255, 255)

font = cv2.FONT_HERSHEY_SIMPLEX

text_AH = "AH: Lock Rectangle"
text_AV = "AV: Strike Zone"
text_AK = "AK: Camera FoV"

class_name = 'iha'

image_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/data/inference/images/test-photo-back.jpeg"
model_path = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/models/trained/best.pt"
save_dir = "/Users/yasinyldrm/Coding/Python-PyCharm/UAV/runs/detect/results/images"
os.makedirs(save_dir, exist_ok=True)

# Generate a unique filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{timestamp}_processed_uav_detection.jpg"
save_path = os.path.join(save_dir, filename)

# Load YOLO model
model = YOLO(model_path)

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not open image.")
    exit()

# Get image dimensions
height, width, _ = image.shape
print(f"Width: {width}, Height: {height}")

# Define yellow rectangle dimensions
yellow_rect_x1 = int(0.25 * width)
yellow_rect_y1 = int(0.10 * height)
yellow_rect_x2 = int(0.75 * width)
yellow_rect_y2 = int(0.90 * height)

# Perform object detection
results = model(image, verbose=False)[0]
boxes = np.array(results.boxes.data.tolist())

# Process the results
for box in boxes:
    x1, y1, x2, y2, score, class_id = box
    x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)

    if results.names[class_id] == class_name and score > confidence_score:
        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color_red, 5)
        score_text = f"{class_name}: {score * 100:.0f}%"
        label_size, baseLine = cv2.getTextSize(score_text, font, 2, 3)
        background_top_left = (x1, y1 - label_size[1] - 5)
        background_bottom_right = (x1 + label_size[0] + 5, y1 + baseLine - 5)
        cv2.rectangle(image, background_top_left, background_bottom_right, color_gray, cv2.FILLED)
        cv2.putText(image, score_text, (x1 + 2, y1 - 3), font, 2, color_white, thickness=3)

        # Additional labels on the image
        cv2.putText(image, text_AH, (x1 - 15, y2 + 50), font, 1.5, color_black, thickness=3)

# Draw the yellow rectangle
cv2.rectangle(image, (yellow_rect_x1, yellow_rect_y1), (yellow_rect_x2, yellow_rect_y2), color_yellow, 6)
cv2.putText(image, text_AV, (yellow_rect_x1 + 25, yellow_rect_y2 - 25), font, 2, color_black, thickness=3)
cv2.putText(image, text_AK, (10, height - 20), font, 1.5, color_black, thickness=3)

# Display timestamp on the image
timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-4]
cv2.putText(image, timestamp, (width - 400, 100), font, 1.6, color_white, thickness=3)

# Save the processed image with a unique name
cv2.imwrite(save_path, image)

# Display the processed image
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Image is saved in {save_path} as {filename}")
