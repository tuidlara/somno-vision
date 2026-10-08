import cv2
import mediapipe as mp
import math

face_detector = mp.tasks.vision.FaceDetector.create_from_model_path(
    "models/blaze_face_short_range.tflite"
)

face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_model_path(
    "models/face_landmarker.task"
)

camera = cv2.VideoCapture(0)

# landmarks para representar os olhos
LEFT_EYE = [33, 133, 159, 145, 158, 153]
RIGHT_EYE = [362, 263, 386, 374, 385, 380]

# calcula distancia entre 2 landmarks
def calculate_distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the camera.")
        break

    # transforma para rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    landmarker_result = face_landmarker.detect(mp_image)

    if landmarker_result.face_landmarks:
        face_landmarks = landmarker_result.face_landmarks[0]

        right_width = calculate_distance(
            face_landmarks[362],
            face_landmarks[263]
        )

        right_height = calculate_distance(
            face_landmarks[386],
            face_landmarks[374]
        )

        right_height_2 = calculate_distance(
            face_landmarks[385],
            face_landmarks[380]
        )

        right_ear = (right_height + right_height_2) / (2 * right_width)

        print(right_width)
        print(right_height)
        print(right_height_2)
        print(right_ear)

        # calcula distancia na horizontal do olho esquerdo
        left_width = calculate_distance(
            face_landmarks[33],
            face_landmarks[133]
        )

        print(left_width)

        left_top = face_landmarks[159]
        left_bottom = face_landmarks[145]

        # calcula a distância na vertical do olho esquerdo
        left_height = calculate_distance(left_top, left_bottom)

        print(left_height)

        left_top_2 = face_landmarks[158]
        left_bottom_2 = face_landmarks[153]

        # calcula a segunda distancia na vertical do olho esquerdo
        left_height_2 = calculate_distance(left_top_2, left_bottom_2)

        print(left_height_2)

        # calcula o EAR do olho esquerdo
        left_ear = (left_height + left_height_2) / (2 * left_width)

        print(left_ear)

        # percorre somente os landmarks selecionados dos olhos
        for index in LEFT_EYE + RIGHT_EYE:
            landmark = face_landmarks[index]

            # converte a posição X de 0-1 para pixels
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    detection_result = face_detector.detect(mp_image)

    if detection_result.detections:
        detection = detection_result.detections[0]

        bounding_box = detection.bounding_box

        cv2.rectangle(
            frame,
            (bounding_box.origin_x, bounding_box.origin_y),
        (
                bounding_box.origin_x + bounding_box.width,
                bounding_box.origin_y + bounding_box.height
        ),
        (0, 255, 0),
        2
    )

    cv2.imshow("Drowsiness Detection - V1", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()