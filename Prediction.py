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
image_all = "testingFootage/football_field.jpg"

results = model.predict(source= image_all, show=True, conf = 0.35, save = True)
#results = model.track(source = highway, show = True, tracker = "bytetrack.yaml")
#print(results)