from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model_car = YOLO('Models/best.pt')
model = YOLO ('Models/MOD_Model.pt')

highway = "testingFootage/prueba.mp4"
tCircle = "testingFootage/traffic_circle_footage.mp4"
germanyFT_1 = "testingFootage/Clip_TopView_720p.mp4"
image_all = "testingFootage/ALL_img.jpg"
image_footbal = "testingFootage/football_field.jpg"

#results = model.predict(source= image_all, show_conf = True, save = True)
#results = model.predict(source= image_footbal, show_conf = True, save = True)
#results = model_car.predict(source= image_all, show_conf = True, save = True)
#results = model_car.predict(source= image_footbal, show_conf = True, save = True)
#results = model.track(source = highway, show = True, tracker = "bytetrack.yaml")
results = model.track(source= "0", save = True, conf = 0.4, show = True)
print(results)