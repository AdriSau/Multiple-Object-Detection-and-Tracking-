
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="D:\ProyectosGithub\Multiple-Object-Detection-and-Tracking-\Sources\data.yaml", epochs=20)  # train the model
