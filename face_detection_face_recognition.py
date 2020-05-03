"""
Module for face detection using face_recognition library.
"""
import os
import time
import cv2
import face_recognition

rect_line_color = (0, 255, 0)
rect_line_width = 2


def detect_face_face_recognition(model, image, save_false_finding=True):
    """
    Detects faces on the received image using face_recognition library.

    :param model: model to be used to detect faces
    :param image: image on which the face should be detected
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    """
    image_copy = image.copy()
    face_locations = face_recognition.face_locations(image_copy, model=model)
    if save_false_finding:
        save_false_findings_face_recognition(face_locations, image_copy)
    return len(face_locations)


def save_false_findings_face_recognition(face_locations, image):
    """
    Saves images that contain exactly one face, but the algorithm either did
    not find any face on them, or found more than one. In latter case
    - rectangles are drawn around every place where the face was found.

    :param face_locations: detections of faces faces found by algorithm
    :param image: image under evaluation
    :return:
    """
    if len(face_locations) == 1:
        return
    if not len(face_locations):
        path_none = "./false_findings/face_recognition/none/"
        if not os.path.exists(path_none):
            os.makedirs(path_none)
        cv2.imwrite(
            path_none + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(face_locations) > 1:
        for face in face_locations:
            cv2.rectangle(
                image,
                (face[0], face[1]),
                (face[2], face[3]),
                rect_line_color,
                rect_line_width)
        path_too_many = "./false_findings/face_recognition/too_many/"
        if not os.path.exists(path_too_many):
            os.makedirs(path_too_many)
        cv2.imwrite(
            path_too_many + str(time.time()) + ".jpg",
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
