import ultralytics
from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
ultralytics.checks()

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")
    model.train(data="D:\cloned_repos\Multiple-Object-Detection-and-Tracking-\Multiple-Object-Detection-and-Tracking-\AUB_DS\data.yaml", epochs=200)
