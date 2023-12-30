from ultralytics import YOLO
import cv2
model = YOLO('best.pt')

highway = "testingFootage/prueba.mp4"
tCircle = "testingFootage/traffic_circle_footage.mp4"
#results = model.predict(source="1", show=True, conf = 0.45, classes = 0)
results = model.track(source = highway,show=True,conf = 0.3, classes = 0, tracker = "bytetrack.yaml")
print(results)
