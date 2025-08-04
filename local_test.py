from ultralytics import YOLO

# Load a model
model = YOLO("weights/last.pt")

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)