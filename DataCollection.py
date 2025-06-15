import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # 0 because we are using device camera
detector = HandDetector(maxHands=1)  # Only 1 hand at a time
offset = 20
imgSize = 300
counter = 0

folder = "C:/Users/User/Desktop/Sign Language Dections/Data/Hello"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the crop stays within bounds
        height, width, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(width, x + w + offset)
        y2 = min(height, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Skip empty crops to avoid resize error
        if imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
