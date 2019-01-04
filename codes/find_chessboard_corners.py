import cv2
import matplotlib.pyplot as plt

nx = 8
ny = 6

file_name = 'calibration_test.png'
img = cv2.imread(file_name)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret:
	cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
	plt.imshow(img)
