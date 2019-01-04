import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt

dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

img = cv2.imread('test_image2.png')
nx = 8
ny = 6


def corners_unwarp(img, nx, ny, mtx, dist):
	# 1) Undistort using mtx and dist
	img2 = cv2.undistort(img, mtx, dist, None, mtx)
	# 2) Convert to grayscale
	gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	# 3) Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	# 4) If corners found:
	if ret == True:
		img2 = cv2.drawChessboardCorners(img2, (nx, ny), corners, ret)
		src = np.float32([corners[0], corners[nx - 1], corners[ny * nx - nx], corners[ny * nx - 1]])
		h, w = img.shape[:2]
		dst = np.float32([[100, 100], [w - 100, 100], [100, h - 100], [w - 100, h - 100]])
		m = cv2.getPerspectiveTransform(src, dst)
		warped = cv2.warpPerspective(img2, m, (w, h), flags = cv2.INTER_LINEAR)
	return warped, m


top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize = 50)
plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)
