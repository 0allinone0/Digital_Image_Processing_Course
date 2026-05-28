import cv2
import numpy as np
import matplotlib.pyplot as plt

#Image Prepare
img1 = cv2.imread('img1.png', 0)
img2 = cv2.imread('img2.png', 0)

h, w =128, 128
img1 = cv2.resize(img1, (w, h)).astype(np.float32) / 255.0
img2 = cv2.resize(img2, (w, h)).astype(np.float32) / 255.0