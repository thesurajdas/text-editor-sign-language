# import python libraries
import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import time
import pyperclip

# Create a classifier object
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
          "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ".", " ", "/"]

# Take size of the image
offset = 20
imgSize = 300
time_delay = 3  # 3 seconds delay
last_time = time.time()

# Function to copy text to clipboard
def copy_text_to_clipboard():
    text = text_editor.get("1.0", "end-1c")
    pyperclip.copy(text)

# Function to clear text
def clear_text():
    text_editor.delete("1.0", tk.END)

# Function to update the video frame
def update_frame():
    global last_time

    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)
    if hands:
        arr = ['*', '*']
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if imgCrop is empty
        if not imgCrop.size:
            return
        imgCropShape = imgCrop.shape

        # Check if the image is portrait or landscape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display the prediction and the letter
        prediction = round(prediction[index]*100, 2)
        print('Percentage: ', prediction)
        print('Letter: ', labels[index])

        # Draw the bounding box and the letter on the video feed
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x + w + offset, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index]+' ('+str(prediction)+')%', (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Check if the letter is the same as the previous letter then add it to the text editor
        current_time = time.time()
        if current_time - last_time > time_delay:
            arr.pop(0)
            arr.append(labels[index])

            if arr[1] != '*' and arr[0] != arr[1]:
                if arr[1] == '/':
                    text_editor.delete(tk.END + "-2c", tk.END)
                else:
                    text_editor.insert(tk.END, arr[1])
            last_time = current_time

    # Display the video feed
    img_output = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    img_output = cv2.resize(img_output, (640, 480))
    photo = ImageTk.PhotoImage(image=Image.fromarray(img_output))
    video_label.config(image=photo)
    video_label.image = photo

    # Update the frame after 10 milliseconds
    root.after(10, update_frame)


# Create a tkinter window
root = tk.Tk()
root.title("Smart Text Editor For Sign Language Users")

# Create a frame for the video feed
video_frame = tk.Frame(root)
video_frame.pack(side=tk.LEFT)

# Create a label to display the video feed
video_label = tk.Label(video_frame)
video_label.pack()

# Create a frame for the text editor
text_frame = tk.Frame(root)
text_frame.pack(side=tk.RIGHT)

# Create a text editor
text_editor = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD)
text_editor.pack()

# Create a "Copy Text" button
copy_button = tk.Button(text_frame, text="Copy Text",
                        command=copy_text_to_clipboard)
copy_button.pack()

# Create a "Clear Text" button
clear_button = tk.Button(text_frame, text="Clear Text", command=clear_text)
clear_button.pack()

# Start updating the video frame
update_frame()

# Run the tkinter main loop
root.mainloop()

# Release the webcam when the application is closed
cap.release()
