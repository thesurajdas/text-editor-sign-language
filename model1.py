import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import time
import pyperclip

# Constants
OFFSET = 20
IMG_SIZE = 300
TIME_DELAY = 3  # 3 seconds delay

# Create a classifier object
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
          "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ".", " ", "/"]

# Initialize tkinter
root = tk.Tk()
root.title("Smart Text Editor For Sign Language Users")

# Create video frame
video_frame = tk.Frame(root)
video_frame.pack(side=tk.LEFT)
video_label = tk.Label(video_frame)
video_label.pack()

# Create text frame
text_frame = tk.Frame(root)
text_frame.pack(side=tk.RIGHT)
text_editor = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
text_editor.pack()

# Function to copy text to clipboard
def copy_text_to_clipboard(text):
    pyperclip.copy(text)

# Function to clear text editor
def clear_text_editor():
    text_editor.delete("1.0", tk.END)

# Initialize variables
last_time = time.time()
arr = ['*', '*']

# Main loop
while True:
    # Read frame from webcam
    success, img = cap.read()
    img_output = img.copy()

    # Detect hands
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop and resize hand region
        img_crop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
        if img_crop.size != 0:
            img_resize = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
            
            # Get prediction from classifier
            prediction, index = classifier.getPrediction(img_resize, draw=False)

            # Update text editor based on gesture
            current_time = time.time()
            if current_time - last_time > TIME_DELAY:
                arr.pop(0)
                arr.append(labels[index])

                if arr[1] != '*' and arr[0] != arr[1]:
                    if arr[1] == '/':
                        clear_text_editor()
                    else:
                        text_editor.insert(tk.END, arr[1])
                last_time = current_time

    # Display the video feed
    img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
    img_output = cv2.resize(img_output, (640, 480))
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_output))
    video_label.config(image=photo)
    video_label.image = photo

    # Update the tkinter window
    root.update()

# Release the webcam when the application is closed
cap.release()

# changes are made from line 73-83 gestures are basically updated
