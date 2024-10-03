import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_nemo_mask(img):
	rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	hsv_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2HSV)

	light_orange = (5, 100, 120)
	dark_orange = (20, 255, 255)
	light_white = (35, 0, 160)
	dark_light = (255, 160, 255)

	mask1 = cv.inRange(hsv_img, light_orange, dark_orange)
	mask2 = cv.inRange(hsv_img, light_white, dark_light)
	mask = np.array((mask1 + mask2) > 0, dtype=mask1.dtype)

	kernel = np.ones((9, 9), np.uint8)
	mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

	ROI = np.zeros(mask.shape, np.uint8)
	ROI[0:166, 50:200] = 1
	mask = mask * ROI

	return mask

