"""
Module for face detection using Dlib.
"""
from __future__ import division

import os
import time
import cv2
import dlib
import numpy as np
from dlib import mmod_rectangle

predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

rect_line_color = (0, 255, 0)
rect_line_width = 2


def detect_face_and_landmarks_dlib(
        detector, image, save_false_finding=True, in_height=300,
        location="dlib"):
    """
    Detects faces on the received image using received detector.

    :param detector: detector to detect faces
    :param image: image on which the face should be detected
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    :param in_height: height of image inserted into the classifier
    :param location: folder name where false findings shall be saved
    :return: 1 if at least one face was found, 0 otherwise
    """
    image_copy = image.copy()
    image_height = image_copy.shape[0]
    image_width = image_copy.shape[1]
    in_width = int((image_width / image_height) * in_height)
    scale = image_height / in_height

    image_small = cv2.resize(image_copy, (in_width, in_height))
    image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
    face_rects = detector(image_small, 0)
    if save_false_finding:
        save_false_findings_dlib(face_rects, image_copy, scale, location)
    return len(face_rects)


def save_false_findings_dlib(face_rectangles, image, scale, location):
    """
    Saves images that contain exactly one face, but the algorithm either did
    not find any face on them, or found more than one. In latter case
    - rectangles are drawn around every place where the face was found.

    :param face_rectangles: rectangles around faces found by algorithm
    :param image: image under evaluation
    :param scale: scale factor by which the image size was decreased before
            feeding to search algorithm
    :param location: folder name where false findings shall be saved
    """
    if len(face_rectangles) > 1:
        for face_r in face_rectangles:

            if isinstance(face_r, mmod_rectangle):
                if face_r.confidence < 0.5:
                    continue
                # cnn version of dlib returns dlib.mmod_rectangle
                # hog version returns rectangle
                face_r = face_r.rect
            shape = predictor(image, face_r)
            # shape are dlib points - change it to normal vector of tuples
            vec = []
            for i in range(68):
                vec.append((int(shape.part(i).x * scale),
                            int(shape.part(i).y * scale)))
            for p1, p2 in zip(vec, vec[1:]):
                cv2.line(image, p1, p2, rect_line_color, rect_line_width)
        path_too_many = os.path.join(
            "./false_findings", location, "too_many/")
        if not os.path.exists(path_too_many):
            os.makedirs(path_too_many)
        cv2.imwrite(
            path_too_many + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not len(face_rectangles):
        path_none = os.path.join("./false_findings", location, "none/")
        if not os.path.exists(path_none):
            os.makedirs(path_none)
        cv2.imwrite(
            path_none + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
