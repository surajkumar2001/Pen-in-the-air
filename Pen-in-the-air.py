# Pen-in-the-air
# main file

import os
import htm
import cv2
import numpy as np

# --------------------------------#
brushThickness: int = 10
eraserThickness: int = 50
drawColor = (0, 0, 0)
SelectedColor: str = 'Black'
# --------------------------------#


# Storing the folder path of Header images in a list
folder_path = "Header images"
myList = os.listdir(folder_path)
print(myList)

over_lay_list = []

for imPath in myList:
    image = cv2.imread(f'{folder_path}/{imPath}')
    over_lay_list.append(image)
print(len(over_lay_list))

header = over_lay_list[0]

# Capturing the image and setting the dimensions
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(3, 1280)  # width property identifier 3
cap.set(4, 720)  # height property identifier 4

xp, yp = 0, 0
imgCanvas = np.zeros([720, 1280, 3], dtype=np.uint8)

count = 0
while True:
    # 0. Detecting hand using detector
    detector = htm.handDetector()

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=True)

    if len(lm_list) != 0:
        #    print(lm_list)

        # tip of index and middle finger
        x1, y1 = lm_list[0][8][1:]
        x2, y2 = lm_list[0][12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            count += 1
            print("Selection mode", count)

            # checking for the click
            if y1 < 125:
                if 180 < x1 < 330:
                    print("Eraser Selected")
                    SelectedColor = 'Eraser'
                    header = over_lay_list[0]
                    drawColor = (0, 0, 0)
                elif 390 < x1 < 540:
                    print("Blue Pen Selected")
                    SelectedColor = 'Blue'
                    header = over_lay_list[1]
                    drawColor = (255, 0, 0)
                elif 580 < x1 < 720:
                    print("Red Pen Selected")
                    SelectedColor = 'Red'
                    header = over_lay_list[2]
                    drawColor = (0, 0, 255)
                elif 780 < x1 < 920:
                    print("Green Pen Selected")
                    SelectedColor = 'Green'
                    header = over_lay_list[3]
                    drawColor = (0, 255, 0)
                elif 970 < x1 < 1110:
                    print("Yellow Pen Selected")
                    SelectedColor = 'Yellow'
                    header = over_lay_list[4]
                    drawColor = (0, 255, 255)
                elif 1140 < x1 < 1280:
                    print("White Pen Selected")
                    SelectedColor = 'White'
                    header = over_lay_list[5]
                    drawColor = (255, 255, 255)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If drawing mode - Index finger is up
        if not (fingers[1] and fingers[2]):
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode", SelectedColor)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Uncomment the following line to view the RGB space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Setting the header image
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, 0.25, imgCanvas, 0.75, 0)
    cv2.imshow("Pen in the Air", img)
    if cv2.waitKey(10) == ord('q'):
        break

