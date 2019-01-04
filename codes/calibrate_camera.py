import pickle

import cv2
import matplotlib.pyplot as plt

dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

img = cv2.imread('test_image.png')


def cal_undistort(img, objpoints, imgpoints):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	undist = cv2.undistort(img, mtx, dist, None, mtx)
	return undist


undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize = 50)
plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)
