import ultralytics
from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
ultralytics.checks()

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model.train(data="D:\ProyectosGithub\Multiple-Object-Detection-and-Tracking-\sources_2\data.yaml", epochs=200)  # train the model
