import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# Ask user for voice choice
print("Select voice:\n1. Male\n2. Female")
choice = input("Enter choice (1 or 2): ")

if choice == '1':
    for voice in voices:
        if "male" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
elif choice == '2':
    for voice in voices:
        if "female" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
else:
    print("Invalid choice. Using default voice.")

# Set up camera and model
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:/Users/User/Desktop/Sign Language Dections/model/keras_model.h5",r"C:/Users/User/Desktop/Sign Language Dections/model/labels.txt")
offset = 20
imgSize = 250

labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]
last_index = -1  # To track the last spoken gesture

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the cropping stays within frame boundaries
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

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

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if 0 <= index < len(labels):
            label = labels[index]
            cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 400, y1 - 10), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x1, y1 - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)

            if index != last_index:
                last_index = index
                engine.say(label)
                engine.runAndWait()

        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()