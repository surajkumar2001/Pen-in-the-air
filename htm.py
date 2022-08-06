# HandTrackingModule code modified as per need

import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, complexity=1, detection_con=0.5, trackCon=0.5):
        self.lmList = []
        self.mode = mode
        self.maxHands = maxHands
        self.detection_con = detection_con
        self.trackCon = trackCon
        self.complexity = complexity
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detection_con, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    # to find if there is any hand in the image that we capture
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        img = imgRGB

        return img

    # to find the positions of the ids on fingers and store it in a list
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(xList, default=0), max(xList, default=0)
        ymin, ymax = min(yList, default=0), max(yList, default=0)
        bbox = [xmin, ymin, xmax, ymax]

        if draw:
            cv2.rectangle(img, (int(xmin) - 20, int(ymin) - 20), (int(xmax) + 20, int(ymax) + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    # to track which fingers are open in the iteration
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    # captures the video
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # shows the image
        cv2.imshow("Image", img)
        # wait for a delay of 10 ms after 'q' key is pressed to terminate the window
        if cv2.waitKey(10) == ord('q'):
            break


if __name__ == "__main__":
    main()
