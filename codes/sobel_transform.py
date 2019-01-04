import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('signs_vehicles_xygrad.png')


def abs_sobel_thresh(img, orient = 'x', thresh_min = 0, thresh_max = 255):
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the derivative in x or y given orient = 'x' or 'y'
	sobel = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y')
	# 3) Take the absolute value of the derivative or gradient
	abs_sobel = np.absolute(sobel)
	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
	# 5) Create a mask of 1's where the scaled gradient magnitude
	# is > thresh_min and < thresh_max
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
	# 6) Return this mask as your binary_output image
	binary_output = sxbinary  # Remove this line
	return binary_output


# Run the function
grad_binary = abs_sobel_thresh(image, orient = 'x', thresh_min = 20, thresh_max = 100)
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(grad_binary, cmap = 'gray')
ax2.set_title('Thresholded Gradient', fontsize = 50)
plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)


def mag_thresh(img, sobel_kernel = 3, mag_thresh = (0, 255)):
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
	# 3) Calculate the magnitude
	mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
	# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
	# 5) Create a binary mask where mag thresholds are met
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	binary_output = np.copy(sxbinary)
	return binary_output


mag_binary = mag_thresh(image, sobel_kernel = 3, mag_thresh = (30, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(mag_binary, cmap = 'gray')
ax2.set_title('Thresholded Magnitude', fontsize = 50)
plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)


def dir_threshold(img, sobel_kernel = 3, thresh = (0, np.pi / 2)):
	# 1) Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
	# 3) Take the absolute value of the x and y gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
	grad_dir = np.arctan2(abs_sobely, abs_sobelx)
	# 5) Create a binary mask where direction thresholds are met
	binary_output = np.zeros_like(grad_dir)
	binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

	# 6) Return this mask as your binary_output image
	binary_output = np.copy(binary_output)  # Remove this line
	return binary_output


# Run the function
dir_binary = dir_threshold(image, sobel_kernel = 15, thresh = (0.7, 1.3))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize = 50)
ax2.imshow(dir_binary, cmap = 'gray')
ax2.set_title('Thresholded Grad. Dir.', fontsize = 50)
plt.subplots_adjust(left = 0., right = 1, top = 0.9, bottom = 0.)