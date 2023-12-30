from ultralytics import YOLO
import cv2
model = YOLO('best.pt')

#results = model.predict(source="1", show=True, conf = 0.45, classes = 0)
results = model.track(source = "1",show=True,conf = 0.45, classes = 0, tracker = "bytetrack.yaml" )
print(results)
