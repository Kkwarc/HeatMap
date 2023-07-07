"""
Michał Kwarciński
"""

from PIL import Image
import glob
import os


folder_path = "gifData/"  # Zmień na odpowiednią ścieżkę

file_list = os.listdir(folder_path)

# for file_name in file_list:
#     if file_name.endswith(".png"):
#         old_path = os.path.join(folder_path, file_name)
#
#         new_file_name = "000" + file_name[:-4]
#         new_file_name = new_file_name[-5:] + ".png"
#         new_path = os.path.join(folder_path, new_file_name)
#
#         os.rename(old_path, new_path)

png_path = "gifData/*.png"
images = []
start = 0
for file in glob.glob(png_path):
    if file[10:] == "700.png":
        start = 1
    if file[9:] == "1100.png":
        break
    if start == 1:
        image = Image.open(file)
        images.append(image)
print(len(images))
images[0].save("heatMap2.gif", save_all=True, append_images=images[1:], loop=0, duration=16)
