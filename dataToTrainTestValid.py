"""
Michał Kwarciński
"""

import os
import shutil

folder_path = f'Photos/'
try:
    os.mkdir(folder_path+'test')
    os.mkdir(folder_path+'train')
    os.mkdir(folder_path+'valid')
    os.mkdir(folder_path+'test/images')
    os.mkdir(folder_path+'test/labels')
    os.mkdir(folder_path+'train/images')
    os.mkdir(folder_path+'train/labels')
    os.mkdir(folder_path+'valid/images')
    os.mkdir(folder_path+'valid/labels')
except FileExistsError:
    pass


mainlist = os.listdir(folder_path)
for mainfilename in os.listdir(folder_path):
    if mainfilename.endswith('.py'):
        continue
    if mainfilename.endswith('.jpg') or mainfilename.endswith('.txt'):
        continue
    print(mainfilename)
    newPath = folder_path+mainfilename+'/'
    i = 0
    list = os.listdir(newPath)
    for filename in os.listdir(newPath):
        if not filename.endswith('.jpg'):
            continue
        jpg = newPath + filename[:-4] + ".jpg"  #  + name + "_"
        txt = newPath + filename[:-4] + ".txt"
        print(jpg)
        print(txt)
        if filename == "classes.txt" or filename == ".jpg" or filename == ".txt":
            continue
        elif i == 5:
            new_name = f'{mainfilename}_{filename}'
            shutil.copy(jpg, os.path.join(folder_path, 'valid/images', new_name))
            shutil.copy(txt, os.path.join(folder_path, 'valid/labels', new_name[:-4] + ".txt"))
            i += 1
            continue
        elif i == 6:
            new_name = f'{mainfilename}_{filename}'
            shutil.copy(jpg, os.path.join(folder_path, 'test/images', new_name))
            shutil.copy(txt, os.path.join(folder_path, 'test/labels', new_name[:-4] + ".txt"))
            i = 0
            continue
        else:
            new_name = f'{mainfilename}_{filename}'
            shutil.copy(jpg, os.path.join(folder_path, 'train/images', new_name))
            shutil.copy(txt, os.path.join(folder_path, 'train/labels', new_name[:-4] + ".txt"))
            i += 1
