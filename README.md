# Multiple object detection and trcking: Yolov8 bytetrack and DeepSORT implementation

## Project Description
This project aims to build a system capable of recognizing different objects and tracking their position, in addition to providing relevant data about them.

The training of the model as well as a brief tracking of them is also provided in this repository. 

In this project we find 2 divided sections:

- Implementation of Yolov8 with the built-in tracker: `Yolo_Prediction.py`
- Implementation of Yolov8 with the DeepSORT algorithm: `main.py`
## How to install and run the proyect
This project is developed on python 3.11 and under a conda virtual environment.For the correct operation it is recommended to install anaconda and import the enviroment provided with the following name `MODT_env.yml`.

As for the DeepSORT algorithm we will use a slightly modified fork to run under python 3.11 from the following repository: <a href="https://github.com/computervisioneng/deep_sort)">Link</a>.

The modified version (Python 3.11) is the following: <a href="https://github.com/AdriSau/deep_sort">Link</a>.To use this version we must clone it with the following command:

```

git clone https://github.com/AdriSau/deep_sort.git

```
