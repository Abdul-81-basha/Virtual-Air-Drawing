import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam!")
    exit()

print("Eye Detection Started!")
print("Press Q to exit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to capture webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = face_mesh.process(rgb_frame)

    status = "NO FACE DETECTED"

    if results.multi_face_landmarks:

        status = "FACE DETECTED - EYES TRACKING"

        for face_landmarks in results.multi_face_landmarks:

            # Draw facial landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )

    # Display status
    color = (0, 255, 0) if results.multi_face_landmarks else (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.imshow("Virtual Air Drawing - Eye Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
