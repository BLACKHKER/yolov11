"""
@Author  ：Zhao
@Date    ：2024/12/19 16:20
@File    ：train.py
@Description: TODO
@Version 1.0 
"""
from ultralytics import YOLO

# Load a model
model = YOLO("./weights/yolo11s.pt")

# Train the model
train_results = model.train(
    data="signature.yaml",  # path to dataset YAML
    epochs=10,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    workers=0,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()