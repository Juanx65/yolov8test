from ultralytics import YOLO
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_directory,"datasets/data.yaml")
model = YOLO('model/yolov8m.pt')

model.train(data=data_path, epochs=50, imgsz=640, batch=16, workers=8)
