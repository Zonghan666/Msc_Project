import cv2
import numpy as np
import skfuzzy as fuzz

def remove_label(img_path):
    img = cv2.imread(img_path, 0)
    ret, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    # smooth = cv2.GaussianBlur(thresh,(5,5), 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_area = np.array([cv2.contourArea(contour) for contour in contours])
    idx = np.argmax(contours_area)

    mask = np.zeros_like(img)
    mask = cv2.drawContours(mask, [contours[idx]], -1, 255, -1)
    mask = (mask > 0)
    result = img * mask

    return result


def remove_pectoral_muscle(img_path):
    img = remove_label(img_path)
    reshape = img.reshape([img.shape[0] * img.shape[1], 1]) / 255
    data = reshape.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data, 3, 2, error=0.005, maxiter=1000, init=None)
    a = np.argmax(u, axis=0)
    result = a.reshape(img.shape[0], img.shape[1])

    cnt_cluster = []
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
    epsilon = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    mask = np.zeros_like(img)
    mask = cv2.drawContours(mask, [approx], -1, 255, -1)

    mask = ((mask > 0) == False)
    result_img = img * mask

    return result_img
