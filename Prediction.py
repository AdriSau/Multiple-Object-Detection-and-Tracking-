from ultralytics import YOLO
import cv2
#model = YOLO('best.pt')
model = YOLO ('2000_img.pt')

highway = "testingFootage/prueba.mp4"
tCircle = "testingFootage/traffic_circle_footage.mp4"
germanyFT_1 = "testingFootage/Video_1_germany.MOV"
germanyFT_2 = "testingFootage/Video_2_germany.MOV"
germanyFT_3 = "testingFootage/Video_3_germany.MOV"

#results = model.predict(source="1", show=True, conf = 0.45, classes = 0)
results = model.track(source = highway,show=True,conf = 0.25, tracker = "bytetrack.yaml")
print(results)
