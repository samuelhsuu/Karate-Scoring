import cv2
from ultralytics import YOLO
import time
from collections import deque, defaultdict

# Load model weights, human_body_detector or hand_detector
model = YOLO("dataset/detector v2.pt")
# Open camera, can also use filepath for video
cap = cv2.VideoCapture("test.mp4")
if not cap.isOpened():
  print("Error: could not open camera")
  exit()

# Objects with confidence <0.3 are filtered out
CONFIDENCE_THRESHOLD = .3
prev_time = time.time()

# Running inference on every other frame for optimization
SKIP_FRAMES = 2
frame_count = 0
last_results = [] # previous frame detections

fps_history = deque(maxlen=30) # Double ended queue of the last 30 frames

font = cv2.FONT_HERSHEY_SIMPLEX

# Draw a info panel on rhs
def draw_sidebar(frame, fps, detection_counts, total_objects):
  h, w = frame.shape[:2] # Unpack first 2 indexes of frame.shape for height and width
  panel_w = 150

  # Background
  overlay = frame.copy()
  cv2.rectangle(overlay, (w-panel_w, 0), (w, h), (15, 15, 25), -1)
  cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

  # Divider
  cv2.line(frame, (w-panel_w, 0), (w-panel_w, h), (60, 60, 80), 1)

  x = w - panel_w + 10
  y = 30
  LINE_H = 26

  # Title
  cv2.putText(frame, "Menu", (x,y), font, 0.5, (100, 100, 140), 1)
  y+=LINE_H

  # FPS
  fps_color = (255, 255, 255)
  if(fps<10):
    fps_color = (0, 0, 255)
  elif(fps>15):
    fps_color = (0, 255, 0)
  cv2.putText(frame, f"FPS {fps:.1f}", (x,y), font, 0.55, fps_color, 1)
  y+=LINE_H + 6

  # Divider
  cv2.line(frame, (x,y), (w-10, y), (60, 60, 80), 1)
  y+=12

  # Per-class counts
  cv2.putText(frame, "Class        Count", (x,y), font, 0.42, (80, 80, 110), 1)
  y+=LINE_H

  for class_name, count in sorted(detection_counts.items(), key = lambda item: -item[1]):
    text = f"{class_name[:12]:<12} { count }"
    cv2.putText(frame, text, (x,y), font, 0.48, (200, 200, 220), 1)
    y+=LINE_H
    if y>h-40:
      break
  # Total
  cv2.putText(frame, f"Total: {total_objects}", (x, h-20), font, 0.5, (100, 200, 255), 1)

while True:
  # Frame is a 3d array containing image data
  ret, frame = cap.read()
  # frame = cv2.flip(frame, 1)

  frame_count+=1
  if frame_count % SKIP_FRAMES == 0:
    # run inference on every other frame
    # model is called like a function to analyze the current frame, specify imgsz for optimization
    last_results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

  if not ret:
    print("Error: failed to read frame")
    break

  detection_counts = defaultdict(int)
  total_objects = 0
  #results only really has one item (the current frame)
  for result in last_results:

    #each box has all detections >0.4 confidence
    for box in result.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      confidence = float(box.conf[0])
      class_id = int(box.cls[0])
      label = model.names[int(box.cls[0])]
      detection_counts[label] += 1
      total_objects += 1

      cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,100), 1)

      text = f"{label} {confidence:.0%}"
      (tw,th),_ = cv2.getTextSize(text, font, 0.6, 2)
      cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6,y1), (0,255,100), -1)

      cv2.putText(frame, text, (x1+3, y1-4), font, 0.6, (0,0,0), 2)

  curr_time = time.time()
  fps_history.append(curr_time - prev_time)
  prev_time = curr_time

  avg_fps = 1.0/(sum(fps_history) / len(fps_history)) # Average the frames
  fps_color = (50, 200, 255)
  if(avg_fps<10):
    fps_color = (255, 0, 0)
  elif(avg_fps>25):
    fps_color = (0, 255, 0)
  draw_sidebar(frame, avg_fps, detection_counts, total_objects) # draw the sidebar

  cv2.imshow("Object Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
cap.release()
cv2.destroyAllWindows()