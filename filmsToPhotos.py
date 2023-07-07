"""
Autor: Michał Kwarciński
"""

import cv2
import os

folder_path = 'Films'

frames = 0
for filename in os.listdir(folder_path):
    cap = cv2.VideoCapture(f"Films/{filename}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames += frame_count
    print(frame_count)
print(frames)

jcap = 60

for filename in os.listdir(folder_path):
    i = 0
    j = 0
    print(filename)
    cap = cv2.VideoCapture(f"Films/{filename}")
    os.mkdir(f'Photos/{filename[:-4]}')
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    while True:
        success, img = cap.read()
        if not success:
            break
        if i == 0 or j > jcap:
            height, width = img.shape[:2]
            scale_percent = 50  # zmniejsz skalę obrazu o 50 procent
            new_width = int(width * scale_percent / 100)
            new_height = int(height * scale_percent / 100)
            new_size = (new_width, new_height)
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

            formatted_string = f"Photos/{filename[:-4]}/" + "{:04d}".format(i) + ".jpg"
            cv2.imwrite(formatted_string, img)
            print(f'Saved: {i}/{int(frame_count / jcap)}')
            i += 1
            j = 0
        j += 1