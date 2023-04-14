from ultralytics import YOLO

model = YOLO('weights/best.pt')

model.track(source='Data1.mp4',show=True)