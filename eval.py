from ultralytics import YOLO
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_directory,"datasets/data.yaml")
model = YOLO('weights/best.pt')

#model.track(source='Data1.mp4',show=True, stream=False, verbose=False) # detectar en el video de testeo

model.val(data=data_path) # evaluar