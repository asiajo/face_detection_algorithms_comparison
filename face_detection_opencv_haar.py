"""
Module for face detection using OpenCV Haar Cascades classifier.
"""
from __future__ import division

import os
import time
import cv2

LBFmodel = "models/lbfmodel.yaml"

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel)

rect_line_color = (0, 255, 0)
rect_line_width = 2


def detect_face_open_cv_cascade_with_landmarks(
        classifier, image, save_false_finding=True,
        location="cv_haar"):
    """
    Detects faces on the received image using received classifier.

    :param classifier: classifier to detect faces
    :param image: image on which the face should be detected
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    :param location: folder name where false findings shall be saved
    :return: 1 if at least one face was found, 0 otherwise
    """
    image_copy = image.copy()
    image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(image_gray)
    if save_false_finding:
        save_false_findings_cv_haar(faces, image_copy, location)
    return len(faces)


def save_false_findings_cv_haar(face_rectangles, image,  location):
    """
    Saves images that contain exactly one face, but the algorithm either did
    not find any face on them, or found more than one. In latter case
    - rectangles are drawn around every place where the face was found.

    :param face_rectangles: rectangles around faces found by algorithm
    :param image: image under evaluation
    :param location: folder name where false findings shall be saved
    """
    if len(face_rectangles) > 1:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, landmarks = landmark_detector.fit(image_gray, face_rectangles)

        for landmark in landmarks:
            for x, y in landmark[0]:
                cv2.circle(image, (x, y), 1, rect_line_color, rect_line_width)
        path_too_many = os.path.join(
            "./false_findings", location, "too_many/")
        if not os.path.exists(path_too_many):
            os.makedirs(path_too_many)
        cv2.imwrite(path_too_many + str(time.time()) + ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not len(face_rectangles):
        path_none = os.path.join("./false_findings", location, "none/")
        if not os.path.exists(path_none):
            os.makedirs(path_none)
        cv2.imwrite(path_none + str(time.time()) + ".jpg",
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
