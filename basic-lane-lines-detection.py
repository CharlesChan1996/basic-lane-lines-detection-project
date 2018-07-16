import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
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
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def Lane_finding(img, kernel_size=5, low_threshold=50, high_threshold=150, rho=2,
                 theta=np.pi / 180, threshold=15, min_line_len=60, max_line_gap=30):
    imshape = img.shape
    gray_img = grayscale(img)
    blur_img = gaussian_blur(gray_img, kernel_size)
    edges = canny(blur_img, low_threshold, high_threshold)
    vertics = np.array([[(0, imshape[0]), (imshape[1] / 2.0 - 20, imshape[0] * 0.6),
                         (imshape[1] / 2.0 + 20, imshape[0] * 0.6), (imshape[1], imshape[0])]], dtype=np.int32)
    edges = region_of_interest(edges, vertics)
    line_img = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)
    return line_img


input_path = '/home/charleschan/PycharmProjects/basic-lane-lines-detection-project/test_images'
output_path = '/home/charleschan/PycharmProjects/basic-lane-lines-detection-project/processed/'

test_images = os.listdir(input_path)
for i in test_images:
    path = input_path + '/' + i
    image = mpimg.imread(path)
    processed_image = Lane_finding(image)
    path = output_path + '/' + i
    mpimg.imsave(path, processed_image, format='jpg')


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image where lines are drawn on lanes)
    line_img = Lane_finding(image)
    result = weighted_img(line_img, image)
    return result


white_output = '/home/charleschan/PycharmProjects/basic-lane-lines-detection-project/test_video_out/solidWhiteRight.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("/home/charleschan/PycharmProjects/basic-lane-lines-detection-project/test_video/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
