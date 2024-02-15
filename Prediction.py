from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#model = YOLO('best.pt')
model = YOLO ('2000_img.pt')

highway = "testingFootage/prueba.mp4"
tCircle = "testingFootage/traffic_circle_footage.mp4"
germanyFT_1 = "testingFootage/Clip_TopView_720p.mp4"
germanyFT_2 = "testingFootage/Video_2_germany.MOV"
germanyFT_3 = "testingFootage/Video_3_germany.MOV"

#results = model.predict(source="1", show=True, conf = 0.45, classes = 0)
results = model.track(source = germanyFT_1, show = True, tracker = "bytetrack.yaml")
#print(results)