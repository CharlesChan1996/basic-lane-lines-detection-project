import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import image
import numpy as np
import cv2
import math

image = mpimg.imread('test_images/solidWhiteRight.jpg')

print('This is the original image:', image.shape, '\n')
print("Save in \"processed\" directory")

mpimg.imsave('processed/solidWhiteRight.png', image)


def greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_to_detect(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_imge = cv2.bitwise_and(img, mask)
    return masked_imge


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    left_slope = []
    right_slope = []
    left_center = []
    right_center = []
    left_len = []
    right_len = []
    slope_eps = 0.35

    def length(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            center = (x1 + x2) / 2, (y1 + y2) / 2

            if slope < -slope_eps:
                left_slope.append(slope)
                left_center.append(center)
                left_len.append(length(x1, y1, x2, y2))
            elif slope > slope_eps:
                right_slope.append(slope)
                right_center.append(center)
                right_len.append(length(x1, y1, x2, y2))
    left_slope = np.array(left_slope)
    right_slope = np.array(right_slope)
    start_y = img.shape[0] * 0.6
    end_y = img.shape[0]
    if len(left_slope > 0):
        left_cnt = 0
        left_center_mean = 0, 0
        left_slope_mean = 0
        for i in range(len(left_slope)):
            left_center_mean = left_center_mean[0] + left_len[i] * left_center[i][0], left_center_mean[1] + left_len[
                i] * left_center[i][1]
            left_slope_mean += left_slope[i] * left_len[i]
            left_cnt += left_len[i]
        if left_cnt > 0:
            left_slope_mean = np.mean(left_slope)
            left_center_mean = np.mean([c[0] for c in left_center]), np.mean([c[1] for c in left_center])
            left_start = int((start_y - left_center_mean[1]) / left_slope_mean + left_center_mean[0]), int(start_y)
            left_end = int((end_y - left_center_mean[1]) / left_slope_mean + left_center_mean[0]), int(end_y)
            cv2.line(img, left_start, left_end, color, thickness)

    if len(right_slope > 0):
        right_center_mean = 0, 0
        right_slope_mean = 0
        right_cnt = 0
        for i in range(len(right_slope)):
            right_center_mean = right_center_mean[0] + right_len[i] * right_center[i][0], right_center_mean[1] + \
                                right_len[i] * right_center[i][1]
            right_slope_mean += right_slope[i] * right_len[i]
            right_cnt += right_len[i]
        if right_cnt > 0:
            right_slope_mean = np.mean(right_slope)
            right_center_mean = np.mean([c[0] for c in right_center]), np.mean([c[1] for c in right_center])
            right_start = int((start_y - right_center_mean[1]) / right_slope_mean + right_center_mean[0]), int(start_y)
            right_end = int((end_y - right_center_mean[1]) / right_slope_mean + right_center_mean[0]), int(end_y)
            cv2.line(img, right_start, right_end, color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Build up a pipeline:
# -- greyscale
# -- Gaussian blur
# -- canny
# -- pick the region for detection
# -- Hough_transform and draw lines

def Lane_finding(img, kernel_size=5, low_threshold=50, high_threshold=150, rho=2,
                 theta=np.pi / 180, threshold=15, min_line_len=60, max_line_gap=30):
    imshape = img.shape
    gray_img = greyscale(img)
    gaussian_blurred = gaussian_blur(gray_img, kernel_size)
    canny_img = canny(gaussian_blurred, low_threshold, high_threshold)
    vertices = np.array([[(0, imshape[0]), (imshape[1] / 2.0 - 20, imshape[0] * 0.6),
                          (imshape[1] / 2.0 + 20, imshape[0] * 0.6), (imshape[1], imshape[0])]], dtype=np.int32)
    region_to_detect(canny_img, vertices)
    line_img = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)

    return line_img


