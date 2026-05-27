import cv2
from ultralytics import YOLO
import time
from collections import deque, defaultdict

# Load model weights
karate_model = YOLO("head_body.pt")
# Open camera, can also use filepath for video
cap = cv2.VideoCapture("test.mp4")
if not cap.isOpened():
    print("Error: could not open camera")
    exit()

# Set up for recording video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_out = 15  # video fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
out = cv2.VideoWriter("output.mp4", fourcc, fps_out, (frame_width, frame_height))


# Objects with confidence <0.3 are filtered out
CONFIDENCE_THRESHOLD = 0.5
prev_time = time.time()

# Running inference on every other frame for optimization
SKIP_FRAMES = 1
frame_count = 0
last_results = []  # previous frame detections

fps_history = deque(maxlen=30)  # Double ended queue of the last 30 frames

detect_font = cv2.FONT_HERSHEY_PLAIN
UI_font = cv2.FONT_HERSHEY_SIMPLEX

TOP_K = {
    "blue-fighter": 1,
    "red-fighter": 1,
    "red-glove": 2,
    "blue-glove": 2,
    "red-foot": 2,
    "blue-foot": 2,
    "body" : 2,
    "head" : 2,
}

# Color coding
CLASS_COLORS = {
    "blue-fighter": (255, 180, 0),
    "red-fighter": (0, 0, 180),
    "blue-glove":  (255, 180, 0),
    "red-glove":   (0, 0, 180),
    "blue-foot":   (255, 180, 0),
    "red-foot":    (0, 0, 180),
    "head":        (0, 255, 0),
    "body":        (0, 255, 0),
}

# Filter for 2 athletes
def filter_top_k(results, model, top_k):
    # Group by class name
    class_detections = defaultdict(list)
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            coords = tuple(map(int, box.xyxy[0]))
            class_detections[label].append((conf, coords, box))

    # Sort each class by confidence and keep top K
    filtered = []
    for label, detections in class_detections.items():
        limit = top_k.get(label, 1)
        detections.sort(key=lambda x: x[0])  # Sort by decreasing confidence
        for conf, coords, box in detections[:limit:]:
            filtered.append((label, conf, coords))
    return filtered

# Draw a info panel on rhs
def draw_sidebar(frame, fps, detection_counts, total_objects):
    h, w = frame.shape[:2]  # Unpack first 2 indexes of frame.shape for height and width
    panel_w = 150

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Divider
    cv2.line(frame, (w - panel_w, 0), (w - panel_w, h), (60, 60, 80), 1)

    x = w - panel_w + 10
    y = 30
    LINE_H = 26

    # Title
    cv2.putText(frame, "Menu", (x, y), UI_font, 0.5, (100, 100, 140), 1)
    y += LINE_H

    # FPS
    fps_color = (255, 255, 255)
    if fps < 10:
        fps_color = (0, 0, 255)
    elif fps > 15:
        fps_color = (0, 255, 0)
    cv2.putText(frame, f"FPS {fps:.1f}", (x, y), UI_font, 0.55, fps_color, 1)
    y += LINE_H + 6

    # Divider
    cv2.line(frame, (x, y), (w - 10, y), (60, 60, 80), 1)
    y += 12

    # Per-class counts
    cv2.putText(frame, "Class        Count", (x, y), UI_font, 0.42, (80, 80, 110), 1)
    y += LINE_H
    
    for class_name, count in sorted(
        detection_counts.items(), key=lambda item: -item[1]
    ):
        text = f"{class_name[:12]:<12} { count }"
        cv2.putText(frame, text, (x, y), UI_font, 0.48, (200, 200, 220), 1)
        y += LINE_H
        if y > h - 40:
            break
    # Total
    cv2.putText(
        frame, f"Total: {total_objects}", (x, h - 20), UI_font, 0.5, (100, 200, 255), 1
    )

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "hi", (0,0),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),3)
    frame_count += 1
    if frame_count % SKIP_FRAMES == 0:
        karate_results = karate_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        last_results   = karate_results   # update only, no drawing

    # All drawing happens here, every frame
    detections = filter_top_k(last_results, karate_model, TOP_K)

    detection_counts = defaultdict(int)
    total_objects = 0

    for label, conf, (x1, y1, x2, y2) in detections:
        detection_counts[label] += 1
        total_objects += 1
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 1)

    if not ret:
        print("Error: failed to read frame")
        break

    detection_counts = defaultdict(int)
    total_objects = 0
    for label, conf, coords in detections:
        detection_counts[label] += 1
        total_objects += 1
        x1,y1,x2,y2 = map(int, coords)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLASS_COLORS.get(label, (255, 180, 0)), 1)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, detect_font, 1, 2)
        cv2.rectangle(
            frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), CLASS_COLORS.get(label, (255, 180, 0)), -1
        )

        cv2.putText(frame, text, (x1 + 3, y1 - 4), detect_font, 1, (0, 0, 0), 2)
        
    curr_time = time.time()
    fps_history.append(curr_time - prev_time)
    prev_time = curr_time

    avg_fps = 1.0 / (sum(fps_history) / len(fps_history))  # Average the frames
    fps_color = (50, 200, 255)
    if avg_fps < 10:
        fps_color = (255, 0, 0)
    elif avg_fps > 25:
        fps_color = (0, 255, 0)
    draw_sidebar(frame, avg_fps, detection_counts, total_objects)  # draw the sidebar

    cv2.imshow("Object Detection", frame)
    out.write(frame)  # write to video file

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
