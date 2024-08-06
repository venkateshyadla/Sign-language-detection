import tkinter as tk
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
import math
from PIL import Image, ImageTk

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Constants
offset = 30  # Offset to expand hand bounding box
imgSize = 224  # Size of the square image for gesture prediction
IMAGE_SIZE = (224, 224)  # Image size expected by the model
FPS = 10  # Frames per second for video stream

# Load the trained model for gesture prediction
model = tf.keras.models.load_model('Trained_models/latest_models/keras_model_S.h5')

# Read the class labels from the file
with open('Trained_models/latest_models/labels_W.txt', 'r') as file:
    labels_list = [line.strip() for line in file.readlines()]

def preprocess_image(img):
    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to range [0, 1]
    img_normalized = img_rgb / 255.0
    # Resize the image to match the model's input size
    resized_img = cv2.resize(img_normalized, IMAGE_SIZE)
    return resized_img

def predict_gesture(hand_img):
    preprocessed_img = preprocess_image(hand_img)
    # Make a prediction using the loaded model
    predictions = model.predict(np.array([preprocessed_img]))
    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions)
    return predicted_class_index

def show_frames():
    # Read frame from camera
    success, img = cap.read()
    frame = cv2.flip(img, 1)  # Flip frame horizontally for mirror view
    
    # Detect hands in the frame
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get bounding box coordinates
        
        # Check if the hand is within the image boundaries
        if x - offset >= 0 and y - offset >= 0 and x + w + offset < img.shape[1] and y + h + offset < img.shape[0]:
            canvas = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white canvas
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop hand region
            
            # Resize and center the hand image within the canvas
            aspectRatio = h / w
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
                
            cv2.imshow("Gesture Monitor", canvas)  # Display processed hand image
            predicted_class_index = predict_gesture(canvas)  # Predict gesture class index
            
            # Retrieve the predicted gesture label from the list
            class_label = labels_list[predicted_class_index]
            cv2.putText(frame, class_label, (frame.shape[1] - 200, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 2, cv2.FILLED)
    
    # Convert frame to RGB and display in tkinter window
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    panel.img = img
    panel.config(image=img)
    
    # Schedule next frame update after specified time (milliseconds)
    panel.after(1000 // FPS, show_frames)

def quit():
    # Release camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Create main tkinter window
root = tk.Tk()
root.title("Sign Language Prediction")

# Create label to display video stream
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# Configure window close event to call quit function
root.protocol("WM_DELETE_WINDOW", quit)

# Start displaying video frames
show_frames()

# Start the tkinter main event loop
root.mainloop()