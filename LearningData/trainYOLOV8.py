"""
Michał Kwarciński
"""

from ultralytics import YOLO

# Load the model.
model = YOLO('../Yolo-weights/yolov8n.pt')

# Training. C:/Users/kwarc/PycharmProjects/ASO_PROJEKT/aso_projekt14/LearningData/data.yaml
results = model.train(
    data='C:/Users/kwarc/PycharmProjects/ASO_PROJEKT/aso_projekt14/LearningData/data.yaml',
    imgsz=640,
    epochs=25,
    batch=10,
    name='yolov8n_custom_test_full_video_data')
validation = model.val()
