import cv2
import numpy as np
import mediapipe as mp
import random

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Define colors (BGR format)
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), 
          (0, 255, 255), (255, 255, 255), (0, 0, 0)]  # Last one is black (Eraser)
colorIndex = 0
shape = 'line'  # Default shape
tool = 'brush'  # Default tool

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to access webcam")
    cap.release()
    exit()

# Initialize canvas
canvas = np.zeros_like(frame)
drawing = False
prev_x, prev_y = None, None

def celebration_effect(frame):
    for _ in range(100):
        x, y = random.randint(0, frame.shape[1]), random.randint(0, frame.shape[0])
        cv2.circle(frame, (x, y), random.randint(5, 10), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Resize canvas if needed
    if canvas.shape[:2] != frame.shape[:2]:
        canvas = np.zeros_like(frame)
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        hand_list = []
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
            hand_list.append(landmarks)
            
            index_finger_tip = landmarks[8]
            x, y = index_finger_tip
            
            # Celebration gesture (both thumbs up)
            if len(hand_list) == 2 and all(hand[4][1] < hand[3][1] for hand in hand_list):
                celebration_effect(frame)
                cv2.putText(frame, "Celebration!", (w // 3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                continue
            
            # Color selection
            if y < 50:
                colorIndex = x // (w // len(colors))
            elif 50 < y < 100:
                if x < w // 4:
                    shape = 'circle'
                elif x < w // 2:
                    shape = 'triangle'
                elif x < 3 * w // 4:
                    shape = 'rectangle'
                else:
                    shape = 'line'
            elif 100 < y < 150:
                if x < w // 3:
                    tool = 'brush'
                elif x < (2 * w // 3):
                    tool = 'highlighter'
                else:
                    tool = 'spray'
            else:
                # Drawing logic
                if prev_x is None or prev_y is None:
                    prev_x, prev_y = x, y
                thickness = 20 if colorIndex == len(colors) - 1 else (5 if tool == 'brush' else 15)
                
                if shape == 'line':
                    cv2.line(canvas, (prev_x, prev_y), (x, y), colors[colorIndex], thickness)
                elif shape == 'circle':
                    cv2.circle(canvas, (x, y), 30, colors[colorIndex], -1)
                elif shape == 'triangle':
                    pts = np.array([[x, y - 30], [x - 30, y + 30], [x + 30, y + 30]], np.int32)
                    cv2.polylines(canvas, [pts], isClosed=True, color=colors[colorIndex], thickness=5)
                elif shape == 'rectangle':
                    cv2.rectangle(canvas, (x - 30, y - 30), (x + 30, y + 30), colors[colorIndex], thickness)
                
                prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None
    
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    # UI Elements
    for i, col in enumerate(colors):
        cv2.rectangle(frame, (i * (w // len(colors)), 0), ((i + 1) * (w // len(colors)), 50), col, -1)
    
    cv2.putText(frame, "Circle", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Triangle", (w//4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Rectangle", (w//2, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Line", (3 * w//4, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "Brush", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Highlighter", (w//3, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Spray", (2 * w//3, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if colorIndex == len(colors) - 1 and np.sum(canvas) == 0:
        cv2.putText(frame, "Canvas Cleared!", (w // 3, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        celebration_effect(frame)
    
    cv2.imshow("Virtual Painter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()