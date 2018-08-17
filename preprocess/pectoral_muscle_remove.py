import cv2
import numpy as np
import skfuzzy as fuzz

def remove_label(img_path):
    """
    function to remove label and other noise and reserve the brest part
    :param img_path: path of the processed image
    :return: result n-array cv2 format
    """

    img = cv2.imread(img_path, 0)
    ret, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    # smooth = cv2.GaussianBlur(thresh,(5,5), 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = np.array([cv2.contourArea(contour) for contour in contours])
    idx = np.argmax(contours_area)

    mask = np.zeros_like(img)
    # we just keep the Contours with the largest area, which is normally the brest part
    mask = cv2.drawContours(mask, [contours[idx]], -1, 255, -1)
    mask = (mask > 0)
    result = img * mask

    return result


def remove_pectoral_muscle(img_path):
    """
    function to remove the pectoral muscle
    :param img_path: path of the image
    :return: result_img n-array cv2 format
    """

    img = remove_label(img_path)
    reshape = img.reshape([img.shape[0] * img.shape[1], 1]) / 255
    data = reshape.T
    # apply fuzzy-c-means algorithm
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, 3, 2, error=0.005, maxiter=1000, init=None)
    a = np.argmax(u, axis=0)
    result = a.reshape(img.shape[0], img.shape[1])

    cnt_cluster = []
    # find the pectoral muscle cluster
    for i in range(3):
        result_muscle = (result == i)
        mask = np.zeros_like(img)
        mask[result_muscle] = 255
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_area = np.array([cv2.contourArea(contour) for contour in contours])
        idx = np.argmax(contours_area)
        cnt_cluster.append(contours[idx])

    contours_area = np.array([cv2.contourArea(contour) for contour in cnt_cluster])
    idx = np.argmin(contours_area)
    cnt = cnt_cluster[idx]
    # approximate the raw 'bad' shape
    epsilon = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    mask = np.zeros_like(img)
    mask = cv2.drawContours(mask, [approx], -1, 255, -1)

    mask = ((mask > 0) == False)
    result_img = img * mask

    return result_img


def cut_background(img, ratio=0.8):
    """
    function to cut the useless background (rectangle)
    :param img: n-array cv2 format
    :param ratio: ratio of background that need to be cut
    :return: img n-array cv2 format
    """
    # first we sum over the columns, background should be the columns of sum 0
    img_sum = np.sum(img, axis=0)

    # next we need to check whether the background is on the right of left part of the images
    # we need to standardise the image so that the background is on the right hand side
    if img_sum[0] == 0:
        img_type = 'right'
    elif img_sum[-1] == 0:
        img_type = 'left'

    if img_type == 'right':
        img = img[:, ::-1]
        img_sum = img_sum[::-1]

    # use np.argmin() to find the border of the background
    start = np.argmin(img_sum)
    end = img_sum.shape[0] - 1
    reserved_pixels = int((end - start) * (1 - ratio))
    img = img[:, :start + reserved_pixels]

    if img_type == 'right':
        img = img[:, ::-1]

    return img
