"""
Module for face detection using OpenCV Haar Cascades classifier.
"""
from __future__ import division

import os
import time
import cv2

rect_line_color = (0, 255, 0)
rect_line_width = 2


def detect_face_open_cv_cascade(
        classifier, image, save_false_finding=True, in_height=300):
    """
    Detects faces on the received image using received classifier.

    :param classifier: classifier to detect faces
    :param image: image on which the face should be detected
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    :param in_height: height of image inserted into the classifier
    :return: 1 if at least one face was found, 0 otherwise
    """
    image_copy = image.copy()
    image_height = image_copy.shape[0]
    image_width = image_copy.shape[1]
    in_width = int((image_width / image_height) * in_height)

    scale = image_height / in_height

    image_small = cv2.resize(image_copy, (in_width, in_height))
    image_gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(image_gray)
    if save_false_finding:
        save_false_findings_cv_haar(faces, image_copy, scale)
    return len(faces)


def save_false_findings_cv_haar(face_rectangles, image, scale):
    """
    Saves images that contain exactly one face, but the algorithm either did
    not find any face on them, or found more than one. In latter case
    - rectangles are drawn around every place where the face was found.

    :param face_rectangles: rectangles around faces found by algorithm
    :param image: image under evaluation
    :param scale: scale factor by which the image size was decreased before
            feeding to search algorithm
    """
    if len(face_rectangles) > 1:
        for (x, y, w, h) in face_rectangles:
            cv2.rectangle(
                image,
                (int(x * scale), int(y * scale)),
                (int((x + w) * scale), int((y + h) * scale)),
                rect_line_color,
                rect_line_width)
        path_too_many = "./false_findings/cv_haar/too_many/"
        if not os.path.exists(path_too_many):
            os.makedirs(path_too_many)
        cv2.imwrite(path_too_many + str(time.time()) + ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not len(face_rectangles):
        path_none = "./false_findings/cv_haar/none/"
        if not os.path.exists(path_none):
            os.makedirs(path_none)
        cv2.imwrite(path_none + str(time.time()) + ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
