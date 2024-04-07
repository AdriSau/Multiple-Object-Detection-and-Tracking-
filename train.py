import ultralytics
from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
ultralytics.checks()

if __name__ == '__main__':
    model = YOLO("yolov9s.yaml")
    model.train(data="D:\cloned_repos\Multiple-Object-Detection-and-Tracking-\Multiple-Object-Detection-and-Tracking-\DS_Cars\data.yaml", epochs=50)
