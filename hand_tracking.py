import mediapipe as mp
import cv2 as cv

# Models
# face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# model object
# face_detect = face_detection.FaceDetection(min_detection_confidence=0.8)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
# accessing the webcam
webcam = cv.VideoCapture(0)

webcam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    success, frame = webcam.read()
    if not success:
        break
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # if results.detections
    if results.multi_hand_landmarks:

        # for detection in detections
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_detection(image, detection)
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # points
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # lines
            )
    # Showing thw Frame
    cv.imshow('Video', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# release all windows
webcam.release()
# destroy all. 
cv.destroyAllWindows()

