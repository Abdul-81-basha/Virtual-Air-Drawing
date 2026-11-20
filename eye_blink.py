
import cv2
import mediapipe as mp
import time
import math

# --------------------------------
# MEDIAPIPE FACE MESH
# --------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --------------------------------
# EYE LANDMARKS
# --------------------------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# --------------------------------
# DISTANCE CALCULATION
# --------------------------------

def distance(p1, p2):
    return math.dist(p1, p2)


# --------------------------------
# EYE ASPECT RATIO
# --------------------------------

def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:

        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)

        points.append((x, y))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])

    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0, points

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear, points


# --------------------------------
# CONFIGURATION
# --------------------------------

EAR_THRESHOLD = 0.19

BLINK_MIN_DURATION = 0.05
BLINK_MAX_DURATION = 0.8

DROWSY_DURATION = 1.0
SLEEP_WARNING_DURATION = 5.0

# --------------------------------
# VARIABLES
# --------------------------------

blink_count = 0

eyes_closed = False
closed_start = None

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam not detected!")
    exit()

print("--------------------------------")
print("EYE BLINK DETECTION SYSTEM")
print("--------------------------------")
print("Press Q to exit")

# --------------------------------
# MAIN LOOP
# --------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    status = "NO FACE DETECTED"
    color = (0, 0, 255)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR
        left_ear, left_points = calculate_ear(
            landmarks,
            LEFT_EYE,
            width,
            height
        )

        right_ear, right_points = calculate_ear(
            landmarks,
            RIGHT_EYE,
            width,
            height
        )

        avg_ear = (left_ear + right_ear) / 2

        # Draw eye landmarks
        for point in left_points + right_points:

            cv2.circle(
                frame,
                point,
                2,
                (0, 255, 0),
                -1
            )

        # --------------------------------
        # EYE CLOSURE DETECTION
        # --------------------------------

        if avg_ear < EAR_THRESHOLD:

            if not eyes_closed:

                eyes_closed = True
                closed_start = time.time()

            closed_duration = time.time() - closed_start

            if closed_duration >= SLEEP_WARNING_DURATION:

                status = "POSSIBLE SLEEP - EYES CLOSED"
                color = (0, 0, 255)

            elif closed_duration >= DROWSY_DURATION:

                status = "DROWSINESS WARNING"
                color = (0, 100, 255)

            else:

                status = "EYES CLOSING"
                color = (0, 255, 255)

        else:

            # Eyes reopened
            if eyes_closed:

                closed_duration = time.time() - closed_start

                if BLINK_MIN_DURATION <= closed_duration <= BLINK_MAX_DURATION:

                    blink_count += 1

                eyes_closed = False
                closed_start = None

            status = "ACTIVE - EYES OPEN"
            color = (0, 255, 0)

        # --------------------------------
        # DISPLAY INFORMATION
        # --------------------------------

        cv2.putText(
            frame,
            f"EAR: {avg_ear:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if eyes_closed:

            duration = time.time() - closed_start

            cv2.putText(
                frame,
                f"Closed: {duration:.1f}s",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    else:

        # Reset closure tracking when face disappears
        eyes_closed = False
        closed_start = None

    # --------------------------------
    # STATUS DISPLAY
    # --------------------------------

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.imshow(
        "Virtual Air Drawing - Eye Activity Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------------
# CLEANUP
# --------------------------------

cap.release()
face_mesh.close()
cv2.destroyAllWindows()

print("Eye detection stopped.")
print("Total blinks detected:", blink_count)

