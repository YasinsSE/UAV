# Imports
import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Function to display images with bounding boxes
def show_image_with_boxes(image, results):
    # Convert to RGB (cv2 loads images in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Loop through the results and draw bounding boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract box coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class index
        
        # Draw the bounding box
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add label and confidence
        label = f'{results[0].names[cls]}: {conf:.2f}'
        cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Show the processed image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show()

    # Keep the window open until a key is pressed
    cv2.waitKey(0)

# Main function
if __name__ == "__main__":
    # Load trained model
    model = YOLO('models/best.pt')  # Update the model path if needed

    # Define the directory with test images
    test_images_path = 'data/images/test'  # Path to test images

    # Get a list of all image files in the directory
    image_files = os.listdir(test_images_path)

    # Limit to the first 5 images
    for idx, img_file in enumerate(image_files[:5]):  # Only process 5 images
        img_path = os.path.join(test_images_path, img_file)

        # Read the image using OpenCV
        img = cv2.imread(img_path)
        
        # Run YOLO model on the image
        results = model(img)
        
        # Display the image with bounding boxes
        show_image_with_boxes(img, results)
