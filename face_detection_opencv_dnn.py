"""
Module for face detection using OpenCV deep neural network.
"""
from __future__ import division

import os
import time
import cv2
import numpy as np

landmarknet = cv2.dnn.readNetFromCaffe('./models/landmark_deploy.prototxt',
                                       './models/VanFace.caffemodel')

conf_threshold = 0.7
rect_line_color = (0, 255, 0)
rect_line_width = 2


def detect_face_open_cv_dnn(net, image, save_false_finding=True,
                            location="cv_dnn"):
    """
    Detects faces on the received image using received dnn.

    :param net: deep neural network to detect the face
    :param image: image on which the face should be detected
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    :param location: folder name where false findings shall be saved
    :return: 1 if at least one face was found, 0 otherwise
    """
    image_copy = image.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image_copy, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    count = 0
    list_bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (l, t, r, b) = box.astype("int")  # l t r b

            original_vertical_length = b - t
            t = int(t + original_vertical_length * 0.15)
            b = int(b - original_vertical_length * 0.05)

            margin = ((b - t) - (r - l)) // 2
            l = l - margin if (b - t - r + l) % 2 == 0 else l - margin - 1
            r = r + margin
            refined_box = [l, t, r, b]
            list_bboxes.append(refined_box)
    if save_false_finding:
        save_false_findings_cv_dnn(count, list_bboxes, image_copy, location)
    return count


def save_false_findings_cv_dnn(count, detections, image, location):
    """
    Saves images that contain exactly one face, but the algorithm either did
    not find any face on them, or found more than one. In latter case
    - rectangles are drawn around every place where the face was found.

    :param count: count of faces found by algorithm
    :param detections: detections of faces faces found by algorithm
    :param image: image under evaluation
    :param location: folder name where false findings shall be saved
    """
    if count > 1:

        for bbox in detections:
            list_clm = get_landmarks_caffe(bbox, image)
            for landmark in list_clm:
                for idx, point in enumerate(landmark):
                    cv2.circle(
                        image, point, 2, rect_line_color, rect_line_width)
        path_too_many = os.path.join(
            "./false_findings", location, "too_many/")
        if not os.path.exists(path_too_many):
            os.makedirs(path_too_many)
        cv2.imwrite(
            path_too_many + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not count:
        path_none = os.path.join("./false_findings", location, "none/")
        if not os.path.exists(path_none):
            os.makedirs(path_none)
        cv2.imwrite(
            path_none + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def get_landmarks_caffe(bbox, image):
    """
    Gets the landmarks using caffe DNN.

    :param bbox: bounding box of the face
    :param image: image under evaluation
    :return: list of points of landmarks detected
    """
    lm_caffe_param = 60
    list_clm = []  # caffe landmark list
    l, t, r, b = bbox
    roi = image[t:b + 1, l:r + 1]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    res = cv2.resize(gray_roi,
                     (lm_caffe_param, lm_caffe_param)).astype(np.float32)
    m = np.zeros((lm_caffe_param, lm_caffe_param))
    sd = np.zeros((lm_caffe_param, lm_caffe_param))
    mean, std_dev = cv2.meanStdDev(res, m, sd)
    normalized_roi = (res - mean[0][0]) / (0.000001 + std_dev[0][0])
    blob = cv2.dnn.blobFromImage(normalized_roi, 1.0,
                                 (lm_caffe_param, lm_caffe_param))
    landmarknet.setInput(blob)
    caffe_landmark = landmarknet.forward()
    for landmark in caffe_landmark:
        LM = []
        for i in range(len(landmark) // 2):
            x = landmark[2 * i] * (r - l) + l
            y = landmark[2 * i + 1] * (b - t) + t
            LM.append((int(x), int(y)))
        list_clm.append(LM)
    return list_clm
