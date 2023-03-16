import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

with mp_objectron.Objectron(model_name="Cup", min_tracking_confidence=0.5) as objectron:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Empty camera frame")

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = objectron.process(imgRGB)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    image, detected_object.landmarks_2D, objectron.BOX_CONNECTIONS
                )
                mp_drawing.draw_axis(
                    image, detected_object.rotation, detected_object.translation
                )
        cv2.flip(image, 1)
        cv2.imshow("OpenCV feed", image)

        if cv2.waitKey(5) & 0xFF==ord('q'):
            break

cap.release()

