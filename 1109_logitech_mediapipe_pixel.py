import cv2
import mediapipe as mp
from cvzone.SerialModule import SerialObject
import serial
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
serial1 = serial.Serial("/dev/ttyACM0",9600)

fingers = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                middle = hand_landmarks.landmark[12]
                ring = hand_landmarks.landmark[16]
                pinky = hand_landmarks.landmark[20]

                # 손의 바운딩 박스 계산
                min_x = min(thumb.x, index.x, middle.x, ring.x, pinky.x)
                max_x = max(thumb.x, index.x, middle.x, ring.x, pinky.x)
                min_y = min(thumb.y, index.y, middle.y, ring.y, pinky.y)
                max_y = max(thumb.y, index.y, middle.y, ring.y, pinky.y)

                # 손의 영역의 픽셀 수 계산
                pixel_count = int((max_x - min_x) * image.shape[1] * (max_y - min_y) * image.shape[0])

                # 손짓 판단
                if pixel_count>=1000:
                    serial1.write(b'0')
                    time.sleep(15)
                elif pixel_count < 1000:
                    serial1.write(b'1')
                    time.sleep(30)

                cv2.putText(
                    image, text=f"Pixels: {pixel_count}", org=(10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 0, 255), thickness=3)

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()