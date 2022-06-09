import cv2
from Network.facedetect import find_faces
import numpy as np
import os
import tqdm
import shutil

PATH = 'Dataset/BU4DFE/Testing/'
TARGET = 'Dataset/BU4DFE/Preprocessed/Testing/'

if not os.path.exists(TARGET):
 os.mkdir(TARGET)

FOLDERS = os.listdir(PATH)

for folder in FOLDERS:
 print(f"Processing folder {folder}")
 images = [_ for _ in os.listdir('/'.join([PATH, folder])) if _.endswith('.jpg')]
 gt = [_ for _ in os.listdir('/'.join([PATH, folder])) if _.endswith('.ply')][0]
 if not os.path.exists('/'.join([TARGET, folder])):
  os.mkdir('/'.join([TARGET, folder]))
 
 shutil.copy('/'.join([PATH, folder, gt]), '/'.join([TARGET, folder, gt]))

 for image in tqdm.tqdm(images):
  img = cv2.imread('/'.join([PATH, folder, image]))
  img = find_faces(img)[0]
  img = cv2.resize(img, (256, 256))
  
  cv2.imwrite('/'.join([TARGET, folder, image]), img)
 print(f"Folder {folder} processed\n")
