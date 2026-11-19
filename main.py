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
    print("Error: Webcam not detected!")
    exit()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = None

prev_x, prev_y = 0, 0

brush_color = (255, 0, 255)
brush_size = 5

drawing_mode = False
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

# ==========================================
# 3. MAIN LOOP
# ==========================================

print("Virtual Air Drawing Started!")

while True:

    success, frame = cap.read()

    if not success:
        print("Unable to read webcam!")
        break

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    drawing_mode = False

    # ======================================
    # 4. HAND DETECTION
    # ======================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand_landmarks.landmark

            # Index finger
            index_tip = lm[8]
            index_pip = lm[6]

            # Middle finger
            middle_tip = lm[12]
            middle_pip = lm[10]

            # Thumb
            thumb_tip = lm[4]

            # Coordinates
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # Finger states
            index_up = index_tip.y < index_pip.y
            middle_up = middle_tip.y < middle_pip.y

            thumb_up = thumb_tip.x < lm[3].x

            # ==================================
            # 5. DRAWING MODE
            # ==================================

            if index_up and not middle_up:

                drawing_mode = True

                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    brush_color,
                    cv2.FILLED
                )

                if prev_x == 0 and prev_y == 0:

                    prev_x, prev_y = x, y

                if eraser_mode:

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        (0, 0, 0),
                        brush_size * 5
                    )

                else:

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        brush_color,
                        brush_size
                    )

                prev_x, prev_y = x, y

            else:

                prev_x, prev_y = 0, 0

    else:

        prev_x, prev_y = 0, 0

    # ======================================
    # 6. COMBINE CAMERA AND CANVAS
    # ======================================

    frame = cv2.add(frame, canvas)

    # ======================================
    # 7. PILLOW INTERFACE
    # ======================================

    pil_image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    draw.text(
        (20, 15),
        "VIRTUAL AIR DRAWING",
        fill=(255, 255, 0),
        font=font
    )

    draw.text(
        (20, 55),
        "INDEX FINGER : DRAW",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 85),
        "TWO FINGERS : STOP",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 115),
        "C : CLEAR CANVAS",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 145),
        "S : SAVE DRAWING",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 175),
        "E : ERASER ON/OFF",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 205),
        "1-5 : CHANGE COLORS",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 235),
        "+/- : BRUSH SIZE",
        fill=(255, 255, 255),
        font=font
    )

    draw.text(
        (20, 265),
        "Q : EXIT",
        fill=(255, 255, 255),
        font=font
    )

    # ======================================
    # 8. DISPLAY USING OPENCV
    # ======================================

    frame = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow("Virtual Air Drawing System", frame)

    # ======================================
    # 9. KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    # Clear canvas
    if key == ord('c'):

        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        print("Canvas cleared!")

    # Save drawing
    elif key == ord('s'):

        filename = "drawings/my_drawing.png"

        cv2.imwrite(filename, canvas)

        print("Drawing saved successfully:", filename)

    # Change colors
    elif key in [ord(str(i)) for i in range(1, 6)]:

        selected = int(chr(key))

        brush_color = colors[selected][1]

        eraser_mode = False

        print("Color changed:", colors[selected][0])

    # Toggle eraser
    elif key == ord('e'):

        eraser_mode = not eraser_mode

        print("Eraser:", "ON" if eraser_mode else "OFF")

    # Increase brush size
    elif key in [ord('+'), ord('=')]:

        brush_size = min(brush_size + 2, 30)

        print("Brush size:", brush_size)

    # Decrease brush size
    elif key == ord('-'):

        brush_size = max(brush_size - 2, 1)

        print("Brush size:", brush_size)

    # Exit
    elif key == ord('q'):

        break

# ==========================================
# 10. RELEASE RESOURCES
# ==========================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Virtual Air Drawing Closed!")
