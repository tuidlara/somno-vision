import cv2
import mediapipe as mp

face_detector = mp.tasks.vision.FaceDetector.create_from_model_path(
    "models/blaze_face_short_range.tflite"
)
camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        print("Could not access the camera.")
        break

    #transforma para rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

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