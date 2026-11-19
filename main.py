import cv2
import mediapipe as mp
import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. INITIALIZATION
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = None

# Separate previous positions for each hand and finger
prev_points = {}

brush_color = (255, 0, 255)
brush_size = 5

eraser_mode = False

os.makedirs("drawings", exist_ok=True)

# ==========================================
# 2. COLOR OPTIONS
# ==========================================

colors = {
    1: ("PURPLE", (255, 0, 255)),
    2: ("RED", (0, 0, 255)),
    3: ("GREEN", (0, 255, 0)),
    4: ("BLUE", (255, 0, 0)),
    5: ("WHITE", (255, 255, 255))
}

# Finger landmark positions
finger_data = {
    "INDEX": (8, 6),
    "MIDDLE": (12, 10),
    "RING": (16, 14),
    "PINKY": (20, 18)
}

# ==========================================
# 3. MAIN LOOP
# ==========================================

print("Virtual Air Drawing V1.2 Started!")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    current_points = {}

    gesture_status = "NO HAND DETECTED"

    # ======================================
    # 4. MULTIPLE HAND DETECTION
    # ======================================

    if results.multi_hand_landmarks:

        all_status = []

        for hand_id, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            # ==================================
            # 5. FINGER RECOGNITION
            # ==================================

            raised_fingers = []

            for finger_name, positions in finger_data.items():

                tip_id, pip_id = positions

                tip = lm[tip_id]
                pip = lm[pip_id]

                if tip.y < pip.y:

                    raised_fingers.append(finger_name)

            # ==================================
            # 6. GESTURE CLASSIFICATION
            # ==================================

            finger_count = len(raised_fingers)

            active_eraser = False

            # Four fingers = pause

            if finger_count == 4:

                status = "PAUSED"

            # No fingers = fist eraser

            elif finger_count == 0:

                status = "ERASER"
                active_eraser = True

            # One, two or three fingers = drawing

            else:

                status = "DRAWING"

            all_status.append(
                f"Hand {hand_id + 1}: {status}"
            )

            # ==================================
            # 7. DRAWING WITH MULTIPLE FINGERS
            # ==================================

            if status == "DRAWING":

                for finger_name in raised_fingers:

                    tip_id, pip_id = finger_data[finger_name]

                    tip = lm[tip_id]

                    x = int(tip.x * w)
                    y = int(tip.y * h)

                    point_key = (hand_id, finger_name)

                    current_points[point_key] = (x, y)

                    cv2.circle(
                        frame,
                        (x, y),
                        10,
                        brush_color,
                        cv2.FILLED
                    )

                    if point_key in prev_points:

                        px, py = prev_points[point_key]

                        if eraser_mode:

                            cv2.line(
                                canvas,
                                (px, py),
                                (x, y),
                                (0, 0, 0),
                                brush_size * 5
                            )

                        else:

                            cv2.line(
                                canvas,
                                (px, py),
                                (x, y),
                                brush_color,
                                brush_size
                            )

                    else:

                        cv2.circle(
                            canvas,
                            (x, y),
                            brush_size,
                            brush_color,
                            cv2.FILLED
                        )

            # ==================================
            # 8. FIST ERASER
            # ==================================

            elif active_eraser:

                # Use index fingertip as eraser position
                tip = lm[8]

                x = int(tip.x * w)
                y = int(tip.y * h)

                point_key = (hand_id, "ERASER")

                current_points[point_key] = (x, y)

                cv2.circle(
                    frame,
                    (x, y),
                    brush_size * 5,
                    (0, 0, 255),
                    2
                )

                if point_key in prev_points:

                    px, py = prev_points[point_key]

                    cv2.line(
                        canvas,
                        (px, py),
                        (x, y),
                        (0, 0, 0),
                        brush_size * 5
                    )

            # Pause clears previous positions for this hand
            else:

                pass

        gesture_status = " | ".join(all_status)

    # ======================================
    # 9. UPDATE PREVIOUS POSITIONS
    # ======================================

    prev_points = current_points

    # ======================================
    # 10. COMBINE CAMERA AND CANVAS
    # ======================================

    frame = cv2.add(frame, canvas)

    # ======================================
    # 11. USER INTERFACE
    # ======================================

    pil_image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    interface = [
        "VIRTUAL AIR DRAWING V1.2",
        "GESTURE: " + gesture_status,
        "1-3 FINGERS: DRAW",
        "4 FINGERS: PAUSE",
        "CLOSED FIST: ERASER",
        "C: CLEAR CANVAS",
        "S: SAVE DRAWING",
        "E: KEYBOARD ERASER",
        "1-5: CHANGE COLORS",
        "+/-: BRUSH SIZE",
        "Q: EXIT",
        "BRUSH SIZE: " + str(brush_size)
    ]

    y_position = 15

    for text in interface:

        draw.text(
            (20, y_position),
            text,
            fill=(255, 255, 255),
            font=font
        )

        y_position += 30

    # ======================================
    # 12. DISPLAY
    # ======================================

    frame = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow(
        "Virtual Air Drawing V1.2",
        frame
    )

    # ======================================
    # 13. KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):

        canvas = np.zeros(
            (h, w, 3),
            dtype=np.uint8
        )

        prev_points = {}

        print("Canvas cleared!")

    elif key == ord('s'):

        filename = "drawings/my_drawing.png"

        cv2.imwrite(filename, canvas)

        print("Drawing saved:", filename)

    elif key in [ord(str(i)) for i in range(1, 6)]:

        selected = int(chr(key))

        brush_color = colors[selected][1]

        eraser_mode = False

        print("Color:", colors[selected][0])

    elif key == ord('e'):

        eraser_mode = not eraser_mode

        print("Keyboard eraser:", eraser_mode)

    elif key in [ord('+'), ord('=')]:

        brush_size = min(brush_size + 2, 30)

        print("Brush size:", brush_size)

    elif key == ord('-'):

        brush_size = max(brush_size - 2, 1)

        print("Brush size:", brush_size)

    elif key == ord('q'):

        break

# ==========================================
# 14. RELEASE RESOURCES
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Virtual Air Drawing V1.2 Closed!")
