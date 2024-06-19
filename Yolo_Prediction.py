from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model_car = YOLO('Models/best.pt')
model = YOLO ('Models/Y8_500_HW_Etsisi.pt')

highway = "Datasets_and_testfootage/testingFootage/prueba.mp4"
tCircle = "testingFootage/traffic_circle_footage.mp4"
germanyFT_1 = "testingFootage/Clip_TopView_720p.mp4"
image_all = "testingFootage/ALL_img.jpg"
image_footbal = "testingFootage/football_field.jpg"
etsisi = "Datasets_and_testfootage/testingFootage/ETSISI_AUP.mp4"

#results = model.predict(source= highway, show_conf = True, show = True)
#results = model.predict(source= image_footbal, show_conf = True, save = True)
#results = model_car.predict(source= image_all, show_conf = True, save = True)
#results = model_car.predict(source= image_footbal, show_conf = True, save = True)
results = model_car.track(source = etsisi, show = True, tracker = "bytetrack.yaml", save = True, conf = 0.2)
#results = model.track(source= "0", save = True, conf = 0.4, show = True)
#print(results)