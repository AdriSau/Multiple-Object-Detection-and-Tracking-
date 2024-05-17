import ultralytics
from ultralytics import YOLO
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(torch.cuda.is_available())
print(torch.version.cuda)
ultralytics.checks()

if __name__ == '__main__':

    model = YOLO("yolov8n.yaml")
    metrics = model.val(data="D:\cloned_repos\Multiple-Object-Detection-and-Tracking-\Multiple-Object-Detection-and-Tracking-\AUB_DS\data.yaml")
    metrics.box.map

    model = YOLO("Models/best.pt")
    metrics = model.val(data="D:\cloned_repos\Multiple-Object-Detection-and-Tracking-\Multiple-Object-Detection-and-Tracking-\AUB_DS\data.yaml")
    metrics.box.map

    model = YOLO("Models/Y8_500_HW_Etsisi.pt")
    metrics = model.val(data="D:\cloned_repos\Multiple-Object-Detection-and-Tracking-\Multiple-Object-Detection-and-Tracking-\AUB_DS\data.yaml")
    metrics.box.map

