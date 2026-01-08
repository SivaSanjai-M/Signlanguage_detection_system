import cv2
import mediapipe as mp
import math

# ---------------- INIT ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_x = None

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def fingers_state(lm):
    """
    Returns array of finger states:
    [thumb, index, middle, ring, pinky]
    1 = open, 0 = closed
    """
    state = []

    # Thumb
    state.append(1 if lm[4].x < lm[3].x else 0)

    # Other fingers
    tips = [8, 12, 16, 20]
    pip = [6, 10, 14, 18]
    for t, p in zip(tips, pip):
        state.append(1 if lm[t].y < lm[p].y else 0)

    return state

def detect_sign(lm):
    global prev_x
    fingers = fingers_state(lm)

    # ---------------- RULE ARRAY ----------------
    SIGN_RULES = [
        ([1,1,1,1,1], "HI 🤚"),
        ([0,0,0,0,0], "STOP ✊"),
        ([0,1,1,0,0], "PEACE ✌️"),
        ([1,0,0,0,0], "GOOD 👍"),
        ([0,0,0,0,1], "BAD 👎"),
        ([0,1,1,1,1], "B"),
        ([0,0,0,0,0], "A")
    ]

    # Match static signs
    for rule, label in SIGN_RULES:
        if fingers == rule:
            return label

    # LOVE ❤️ (thumb + index close)
    if distance(lm[4], lm[8]) < 0.05:
        return "LOVE ❤️"

    # BYE 👋 (hand waving)
    if prev_x is not None:
        if abs(lm[0].x - prev_x) > 0.04 and fingers == [1,1,1,1,1]:
            prev_x = lm[0].x
            return "BYE 👋"

    prev_x = lm[0].x
    return "UNKNOWN"

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            text = detect_sign(hand_landmarks.landmark)

    cv2.putText(frame, f"Sign: {text}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Simple Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
