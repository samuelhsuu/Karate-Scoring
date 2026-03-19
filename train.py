from ultralytics import YOLO

if __name__ == '__main__':    
  model = YOLO("yolov8n.pt")
  model.train(
    data = "dataset/data.yaml",
    device = 0,
    workers = 2,
    epochs = 50,
    imgsz = 640,
    batch = 8,
    name="karate_detector"
  )