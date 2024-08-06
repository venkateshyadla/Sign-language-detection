# Import necessary libraries
import cv2
import numpy as np
import math
import time
import os

# Import HandTrackingModule from cvzone
from cvzone.HandTrackingModule import HandDetector

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Constants
offset = 20  # Offset to expand hand bounding box
imgSize = 300  # Size of the square image to be saved
counter = 0  # Counter to keep track of saved images
num_images = 300  # Total number of images to capture
folder = "Signs/a2"  # Folder to save images

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Main loop to capture and process images
while True:
    # Capture frame from camera
    success, img = cap.read()
    
    # Detect hands in the frame
    hands, img = detector.findHands(img, draw=True)
    
    # Process if hands are detected
    if hands:
        hand = hands[0]
        
        # Get bounding box coordinates of the hand
        x, y, w, h = hand['bbox']
        
        # Check if the hand is within the image boundaries
        if x - offset >= 0 and y - offset >= 0 and x + w + offset < img.shape[1] and y + h + offset < img.shape[0]:
            # Create a canvas to place the processed hand image
            canvas = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Crop and resize the hand image to fit the canvas
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w
            
            # Resize and center the hand image within canvas
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                canvas[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                canvas[hGap:hCal + hGap, :] = imgResize
            
            # Display cropped hand and processed canvas
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("Canvas", canvas)
    
    # Display original frame
    cv2.imshow("Image", img)
    
    # Wait for key press
    key = cv2.waitKey(1)
    
    # Save the processed image if 'S' key (key 32) is pressed
    if key == 32:  # 'S' key
        if counter < num_images:
            counter += 1
            # Save image with timestamp as filename
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', canvas)
            print(counter)
        else:
            print(f"Count {num_images} reached. Change to next character.")
            break
    
    # Stop the process if 'q' key is pressed
    if key == ord("q"):
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
