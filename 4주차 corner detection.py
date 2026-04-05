from sklearn.datasets import make_checkerboard
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import numpy as np
import cv2

# make checkerboard
I = np.ones([128, 128]) * 128
for i in range(4):
    for j in range(4):
        if j % 2 == i % 2:
            I[32*(i):32*(i+1), 32*(j):32*(j+1)] = 255

## First Step
# Sobel filter for dx and dy
filter_dx = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])

filter_dy = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])

# Gaussian filter
filter_G = np.array([[1, 2, 1.],
                     [2, 4, 2.],
                     [1, 2, 1]])

filter_G = filter_G / np.sum(filter_G)

I_x = convolve2d(I, filter_dx, mode='same')
I_y = convolve2d(I, filter_dy, mode='same')

#Compute components of covariance matrix at each pixel
I_x2 = I_x ** 2
I_xy = I_x * I_y
I_y2 = I_y ** 2

#Apply a Gaussian filter
gau_I_x2 = convolve2d(I_x2, filter_G, mode="same")
gau_I_xy = convolve2d(I_xy, filter_G, mode="same")
gau_I_y2 = convolve2d(I_y2, filter_G, mode="same")

#Define the Harris matrix at each pixel


det = gau_I_x2 * gau_I_y2 - gau_I_xy **2   #ad-cb in 2x2 matrix
trace = gau_I_x2 + gau_I_y2

R = det - 0.05 * trace ** 2
plt.imshow(R)

# non-maximum suppression
R_nms = np.zeros_like(R)
for i in range(1, R.shape[0] - 1):
    for j in range(1, R.shape[1] - 1):
        if R[i, j] >= R[i - 1:i + 2, j - 1:j + 2].max():
            R_nms[i, j] = R[i, j]

print(np.unique(R_nms))
plt.imshow(R_nms)
plt.show()

plt.imshow(I)
plt.show()