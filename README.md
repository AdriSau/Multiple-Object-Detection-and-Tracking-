# <div align="center">Multiple Object Detection, Tracking and Depth Measurement</div>
 Disclaimer: this project is under development, so there will be constant changes in content and it may not be functional at some point.

## <div align="center">Project Description</div>
The goal of this proyect is to develop a functional algorithm capable of recongnize multiple objects from different classes (3 in this case: cars, people and bikes). The system should also track de different objects and mesure the depth to the camera.

## <div align="center">How to run</div>
The proyect in this stage is not mention to be executed in other devices but if you want to try it you shoud change the following parameters:
* Training source path
* data.yam located in source folder could generate problems while using relative paths, change it if so.

The proyect is developed in a conda enviroment, the training file is ment to be process by the GPU using CUDA but if it is not possible it should automatically change to CPU. If you are planning to use CUDA as it is much faster training you may hace some troubles with the pytorch version so uninstall it and then install it in the official web page, to check CUDA version use "nvcc --version" not "nvidia-smi" and update it if necesary berfore reinstalling pytorch
