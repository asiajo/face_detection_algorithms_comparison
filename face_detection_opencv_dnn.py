"""
Module for face detection using OpenCV deep neural network.
"""
from __future__ import division

import os
import time
import cv2

conf_threshold = 0.7
rect_line_color = (0, 255, 0)
rect_line_width = 2


def detect_face_open_cv_dnn(net, image, save_false_finding=True):
    """
    Detects faces on the received image using received dnn.

    :param net: deep neural network to detect the face
    :param image: image on which the face should be detected
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    :return: 1 if at least one face was found, 0 otherwise
    """
    image_copy = image.copy()
    blob = cv2.dnn.blobFromImage(image_copy, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            count += 1
    if save_false_finding:
        save_false_findings_cv_dnn(count, detections, image_copy)
    return count


def save_false_findings_cv_dnn(count, detections, image):
    """
    Saves images that contain exactly one face, but the algorithm either did
    not find any face on them, or found more than one. In latter case
    - rectangles are drawn around every place where the face was found.

    :param count: count of faces found by algorithm
    :param detections: detections of faces faces found by algorithm
    :param image: image under evaluation
    """
    if count > 1:
        frame_height = image.shape[0]
        frame_width = image.shape[1]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                cv2.rectangle(image, (x1, y1), (x2, y2), rect_line_color,
                              rect_line_width)
        path_too_many = "./false_findings/cv_dnn/too_many/"
        if not os.path.exists(path_too_many):
            os.makedirs(path_too_many)
        cv2.imwrite(
            path_too_many + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not count:
        path_none = "./false_findings/cv_dnn/none/"
        if not os.path.exists(path_none):
            os.makedirs(path_none)
        cv2.imwrite(
            path_none + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
