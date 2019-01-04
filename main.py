import pickle

import cv2
import matplotlib.pyplot as plt

nx = 8
ny = 6

chess_pics_dir = './input/chess_pics'
test_images_dir = './input/test_images'


def get_image(dir_pth, file_name):
	return dir_pth + '/' + file_name


img = cv2.imread(get_image(chess_pics_dir, 'calibration2.jpg'))
dist_pickle = pickle.load('./input/chess_pics/calibration2.jpg')
object_points = dist_pickle["objpoints"]
image_points = dist_pickle["imgpoints"]


def find_chessboard_corners():
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	if ret:
		cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
		plt.imshow(img)


def cal_undistorted(img, object_points, image_points):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
	return undistorted


def calibrate_camera():
	undistorted = cal_undistorted(img, object_points, image_points)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
	f.tight_layout()
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize = 50)
	ax2.imshow(undistorted)
	ax2.set_title('Undistorted Image', fontsize = 50)
	plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)


if __name__ == '__main__':
	find_chessboard_corners()
	calibrate_camera()
