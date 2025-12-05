"""
AirPointer YouTube Gesture Controller — low-latency + visible landmarks
- Processes a small frame for speed, displays a scaled-up frame
- Draws landmarks and connections on the displayed frame (so they are visible)
- Fast model settings, safe action lookups, swipe/mute/thumb gestures
"""

import time
from collections import defaultdict, deque
import webbrowser
import sys

import cv2
from mediapipe.python.solutions import hands as mp_hands
import mediapipe as mp
import numpy as np
import pyautogui

# === Performance settings ===
PROCESS_WIDTH = 320       # width used for MediaPipe processing (small -> faster)
PROCESS_HEIGHT = 240      # height used for processing
DISPLAY_SCALE = 2.0       # how much to scale processed frame for display
MODEL_COMPLEXITY = 0      # 0 = fastest
MIN_DET_CONF = 0.5
MIN_TRACK_CONF = 0.5
PROCESS_EVERY_N_FRAMES = 1  # 1 = every frame

# === App settings ===
COOLDOWN_SECONDS = 0.45
SWIPE_COOLDOWN = 0.9
CAMERA_ID = 0
SHOW_FPS = False
AUTO_OPEN_YOUTUBE = True
YOUTUBE_WAIT_SECONDS = 3

SWIPE_HISTORY = 6
SWIPE_THRESHOLD_FRAC = 0.18
MUTE_HOLD_SECONDS = 0.5

# === Safe wrappers ===
def safe_press(key):
    try:
        pyautogui.press(key)
    except Exception as e:
        print("pyautogui press error:", e)

def safe_hotkey(*keys):
    try:
        pyautogui.hotkey(*keys)
    except Exception as e:
        print("pyautogui hotkey error:", e)

# === Actions mapping ===
ACTIONS = {
    "PLAY": lambda: safe_press('k'),
    "PAUSE": lambda: safe_press('k'),
    "VOLUME_UP": lambda: safe_press('up'),
    "VOLUME_DOWN": lambda: safe_press('down'),
    "NEXT": lambda: safe_hotkey('shift', 'n'),
    "PREV": lambda: safe_hotkey('shift', 'p'),
    "MUTE": lambda: safe_press('m'),
    "SEEK_FWD": lambda: safe_press('l'),
    "SEEK_BACK": lambda: safe_press('j'),
    "THUMB_UP": lambda: safe_press('up'),
    "THUMB_DOWN": lambda: safe_press('down'),
    "FULLSCREEN": lambda: safe_press('f'),
}

# === MediaPipe hands (fast config) ===
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRACK_CONF,
)

# tip ids
TIP_IDS = [4, 8, 12, 16, 20]

def open_youtube_if_needed():
    if not AUTO_OPEN_YOUTUBE:
        return
    try:
        webbrowser.open("https://www.youtube.com")
        print(f"Opening YouTube — waiting {YOUTUBE_WAIT_SECONDS}s...")
        time.sleep(YOUTUBE_WAIT_SECONDS)
    except Exception as e:
        print("Failed to open browser:", e)


def fingers_up_and_coords(hand_landmarks, proc_w, proc_h):
    """Return boolean list [thumb, idx, mid, ring, pinky] and landmark coords in processed frame coordinates."""
    lm = hand_landmarks.landmark
    coords = [(int(l.x * proc_w), int(l.y * proc_h)) for l in lm]

    fingers = [False]*5
    # thumb: compare x between tip and ip
    if coords[TIP_IDS[0]][0] > coords[TIP_IDS[0]-1][0]:
        fingers[0] = True
    for i, tip_id in enumerate(TIP_IDS[1:], start=1):
        if coords[tip_id][1] < coords[tip_id-2][1]:
            fingers[i] = True
    return fingers, coords


def interpret_gesture_basic(fingers):
    thumb, idx, mid, ring, pinky = fingers
    if not any(fingers):
        return "PAUSE"
    if all(fingers):
        return "PLAY"
    if idx and mid and not ring and not pinky:
        return "VOLUME_UP"
    if ring and pinky and not idx and not mid:
        return "VOLUME_DOWN"
    if pinky and not (idx or mid or ring):
        return "NEXT"
    if thumb and idx and mid and ring and not pinky:
        return "PREV"
    return None


def centroid(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))


def draw_landmarks_on_display(display_frame, hand_landmarks):
    """
    Map landmark normalized coords from hand_landmarks to processed pixel coords,
    then scale to display and draw lines & keypoints on display_frame.
    """
    # hand_landmarks.landmark give normalized coords relative to processed frame
    draw_points = []
    for lm in hand_landmarks.landmark:
        x_p = int(lm.x * PROCESS_WIDTH)
        y_p = int(lm.y * PROCESS_HEIGHT)
        x_d = int(x_p * DISPLAY_SCALE)
        y_d = int(y_p * DISPLAY_SCALE)
        draw_points.append((x_d, y_d))

    # draw connections (use list(...) to satisfy type-checkers)
    for (start, end) in list(mp_hands.HAND_CONNECTIONS):
        x1, y1 = draw_points[start]
        x2, y2 = draw_points[end]
        cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # draw points
    for x, y in draw_points:
        cv2.circle(display_frame, (x, y), 4, (0, 0, 255), -1)


def main():
    open_youtube_if_needed()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)

    # try to set camera to a reasonable display size (we'll process smaller)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(PROCESS_WIDTH * DISPLAY_SCALE))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(PROCESS_HEIGHT * DISPLAY_SCALE))

    prev_time = time.time()
    last_trigger = defaultdict(lambda: 0.0)
    last_gesture_name = None

    x_history = deque(maxlen=SWIPE_HISTORY)
    last_swipe_time = 0.0
    fist_start_time = None

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected.")
            break

        # mirror for natural interaction
        frame = cv2.flip(frame, 1)
        orig_h, orig_w = frame.shape[:2]

        # Resize small frame for processing
        proc = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_LINEAR)
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        # Prepare display frame (scaled). We will draw landmarks on this frame.
        if DISPLAY_SCALE != 1.0:
            display_frame = cv2.resize(frame, (int(PROCESS_WIDTH * DISPLAY_SCALE), int(PROCESS_HEIGHT * DISPLAY_SCALE)),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            display_frame = frame.copy()

        process_this_frame = (frame_count % PROCESS_EVERY_N_FRAMES) == 0
        frame_count += 1

        gesture_name = None

        if process_this_frame:
            results = hands.process(proc_rgb)
            mhl = getattr(results, "multi_hand_landmarks", None)
            if mhl:
                hand_landmarks = mhl[0]
                fingers, proc_coords = fingers_up_and_coords(hand_landmarks, PROCESS_WIDTH, PROCESS_HEIGHT)
                gesture_name = interpret_gesture_basic(fingers)

                # draw landmarks on the displayed frame (scaled)
                draw_landmarks_on_display(display_frame, hand_landmarks)

                # compute centroid and append to history for swipe detection
                cx_p, cy_p = centroid(proc_coords)
                x_history.append(cx_p)

                # thumb up/down detection if other fingers down
                other_down = not any([fingers[1], fingers[2], fingers[3], fingers[4]])
                if fingers[0] and other_down:
                    if proc_coords[4][1] < proc_coords[2][1] - 6:
                        gesture_name = "THUMB_UP"
                    elif proc_coords[4][1] > proc_coords[2][1] + 6:
                        gesture_name = "THUMB_DOWN"

                # fist hold for mute
                if not any(fingers):
                    if fist_start_time is None:
                        fist_start_time = time.time()
                    else:
                        if (time.time() - fist_start_time) >= MUTE_HOLD_SECONDS:
                            now = time.time()
                            if now - last_trigger.get("MUTE", 0) > COOLDOWN_SECONDS:
                                last_trigger["MUTE"] = now
                                action = ACTIONS.get("MUTE")
                                if action:
                                    action()
                                last_gesture_name = "MUTE"
                                fist_start_time = None
                else:
                    fist_start_time = None

                # swipe detection (in processed coordinates)
                if len(x_history) >= SWIPE_HISTORY:
                    dx = x_history[-1] - x_history[0]
                    threshold_px = SWIPE_THRESHOLD_FRAC * PROCESS_WIDTH
                    now = time.time()
                    if abs(dx) > threshold_px and (now - last_swipe_time) > SWIPE_COOLDOWN:
                        if dx > 0:
                            action = ACTIONS.get("SEEK_FWD")
                            if action:
                                action()
                            last_gesture_name = "SEEK_FWD"
                        else:
                            action = ACTIONS.get("SEEK_BACK")
                            if action:
                                action()
                            last_gesture_name = "SEEK_BACK"
                        last_swipe_time = now
                        x_history.clear()

                # Normal gesture trigger (with cooldown)
                if gesture_name and gesture_name not in ("SEEK_FWD", "SEEK_BACK", "MUTE", "THUMB_UP", "THUMB_DOWN"):
                    now = time.time()
                    if now - last_trigger[gesture_name] > COOLDOWN_SECONDS:
                        last_trigger[gesture_name] = now
                        action = ACTIONS.get(gesture_name)
                        if action:
                            action()
                        last_gesture_name = gesture_name

                # thumb up/down actions
                if gesture_name == "THUMB_UP":
                    now = time.time()
                    if now - last_trigger["THUMB_UP"] > COOLDOWN_SECONDS:
                        last_trigger["THUMB_UP"] = now
                        action = ACTIONS.get("THUMB_UP")
                        if action:
                            action()
                        last_gesture_name = "THUMB_UP"
                elif gesture_name == "THUMB_DOWN":
                    now = time.time()
                    if now - last_trigger["THUMB_DOWN"] > COOLDOWN_SECONDS:
                        last_trigger["THUMB_DOWN"] = now
                        action = ACTIONS.get("THUMB_DOWN")
                        if action:
                            action()
                        last_gesture_name = "THUMB_DOWN"
            else:
                # no hand: decay history slightly
                if len(x_history) > 0:
                    if len(x_history) > SWIPE_HISTORY // 2:
                        x_history.popleft()

        # Overlays
        disp_h, disp_w = display_frame.shape[:2]
        if last_gesture_name:
            cv2.putText(display_frame, f"Gesture: {last_gesture_name}", (10, disp_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        if SHOW_FPS:
            cur_time = time.time()
            fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
            prev_time = cur_time
            cv2.putText(display_frame, f"FPS: {int(fps)}", (disp_w - 110, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

        cv2.imshow("AirPointer (fast + visible)", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('f'):
            action = ACTIONS.get("FULLSCREEN")
            if action:
                action()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
