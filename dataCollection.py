# import python libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# size of the image
offset = 20
imgSize = 300

# Change folder name for each new gesture
folder = "mydata/A"
counter = 0

# Run this until the program is stopped
while True:
    # Capture frame-by-frame
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # Get hand information
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Check if imgCrop is not empty before processing
        if not imgCrop.size == 0:
            imgCropShape = imgCrop.shape

            # check if the image is portrait or landscape
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Show the diffrent frames
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("Image", imgWhite)

    cv2.imshow("Camera", img)
    key = cv2.waitKey(1)

    # Inside the loop, before saving the image
    if key == ord("s"):
        counter += 1
        filename = f'{folder}/img_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved image {counter} as {filename}")
        
    # Press q to quit
    if key == ord("q"):
        break
# Destroy the window
cv2.destroyWindow('Camera')
