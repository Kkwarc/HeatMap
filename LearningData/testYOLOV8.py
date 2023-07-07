"""
Michał Kwarciński
"""
from ultralytics import YOLO
import os
import cv2

# Load the model.
model = YOLO('C:/Users/kwarc/PycharmProjects/ASO_PROJEKT/aso_projekt14/LearningData/runs/detect/yolov8n_custom_test_full_video_data/weights/best.pt')

folder_path = 'test/images'

for filename in os.listdir(folder_path):
    result = model(f"{folder_path}/{filename}", show=True)
    cv2.waitKey(0)

# cap = cv2.VideoCapture("../Films/testVid1.mp4")  # for video
# while True:
#     success, img = cap.read()
#
#     height, width = img.shape[:2]
#     scale_percent = 50  # zmniejsz skalę obrazu o 50 procent
#
#     new_width = int(width * scale_percent / 100)
#     new_height = int(height * scale_percent / 100)
#     new_size = (new_width, new_height)
#
#     img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
#
#     result = model(img, show=True)
#     cv2.waitKey(1)

# result = model("C:/Users/kwarc/PycharmProjects/ASO_PROJEKT/aso_projekt14/Photos/squareTable7/0011.jpg", show=True)
# cv2.waitKey(0)
