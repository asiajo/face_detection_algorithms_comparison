"""
Module for comparing different methods of face detection. Compares speed and
accuracy.
"""
from __future__ import division

import time
import cv2
from skimage import io
import dlib
import logging
import numpy as np

from face_detection_dlib import detect_face_dlib
from face_detection_opencv_dnn import detect_face_open_cv_dnn
from face_detection_opencv_haar import detect_face_open_cv_cascade
from face_detection_face_recognition import detect_face_face_recognition
from face_detection_and_landmarks_dlib import detect_face_and_landmarks_dlib
from face_detection_deep_face import detect_face_deep_face

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Model files
# OpenCV HAAR
face_cascade = cv2.CascadeClassifier(
    './models/haarcascade_frontalface_default.xml')
# OpenCV DNN supports 2 networks.
# 1. FP16 version of the original caffe implementation ( 5.4 MB )
modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "./models/deploy.prototxt"
net_caffe = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
modelFile = "./models/opencv_face_detector_uint8.pb"
configFile = "./models/opencv_face_detector.pbtxt"
net_tf = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
# DLIB HoG
hogFaceDetector = dlib.get_frontal_face_detector()

# DLIB MMOD
dnnFaceDetector = dlib.cnn_face_detection_model_v1(
    "./models/mmod_human_face_detector.dat")


def run_detection(photos, name, func, model, save_false_finding):
    """
    Common code for calling face detection functions. Calls specified function
    on the set of pictures. Logs number of correctly and incorrectly classified
    images. The log message bases on the assumption, that on every received
    picture exactly one person is present. This is the use case.

    :param photos: list of photos on which face detection should be performed
    :param name: name of the library performing the detection
    :param func: function to be called to perform the detection
    :param model: model to be used to perform the detection
    :param save_false_finding: flag if incorrectly classified images should be
            saved to the disc
    """
    found_correct = 0
    found_false = 0
    t = time.time() * 1000
    for image in photos:
        found = func(model, image, save_false_finding,
                     location="_".join(name.strip().split()))
        found_correct += min(found, 1)
        found_false += int(found > 1)
    time_taken = time.time() * 1000 - t
    amount_of_photos = len(photos)
    l = str(len(str(amount_of_photos)))
    logging.info(
        ("%-26s found faces on %" + l + "d, out of %d pictures in %5d " +
         "milliseconds. On %" + l + "d pictures it made a mistake and found " +
         "more than one face. Achieved accuracy of correct findings: %02d%% " +
         "and mistake rate: %02d%%"
         ) % (name,
              found_correct,
              amount_of_photos,
              time_taken,
              found_false,
              int(found_correct / amount_of_photos * 100),
              int(found_false / amount_of_photos * 100)
              )
    )


def get_minimized_dimensions(w_orig, h_orig):
    """
    Receives dimensions of the existing photo and returns new dimensions, where
    the shorter edge has length [size_short_side] and the longer is calculated
    proportionally to constrain proportions.

    :param w_orig: width of the original image
    :param h_orig: height of the original image
    :return: proportional dimensions, where the smaller one is
           [size_short_side]
    """
    size_short_side = 200
    if w_orig / h_orig < 1:
        return size_short_side, int(size_short_side * h_orig / w_orig)
    else:
        return int(size_short_side * w_orig / h_orig), size_short_side


def main():
    """
    main
    """
    save_false_findings = True

    ic = io.ImageCollection('./samples/*.jpg')
    photos = []
    for image in ic:
        w, h = get_minimized_dimensions(image.shape[1], image.shape[0])
        image_small = np.array(cv2.resize(image, (w, h)))
        photos.append(image_small)
    logging.info("Total amount of photos: %d", len(ic))

    run_detection(
        photos,
        "OpenCV Haar",
        detect_face_open_cv_cascade,
        face_cascade,
        save_false_findings)
    run_detection(
        photos,
        "OpenCV Dnn Caffe",
        detect_face_open_cv_dnn,
        net_caffe,
        save_false_findings)
    run_detection(
        photos,
        "OpenCV Dnn Tf",
        detect_face_open_cv_dnn,
        net_tf,
        save_false_findings)
    run_detection(
        photos,
        "Dlib Hog",
        detect_face_dlib,
        hogFaceDetector,
        save_false_findings)
    run_detection(
        photos,
        "Dlib hog + landmarks",
        detect_face_and_landmarks_dlib,
        hogFaceDetector,
        save_false_findings)
    run_detection(
        photos,
        "Dlib cnn",
        detect_face_dlib,
        dnnFaceDetector,
        save_false_findings)
    run_detection(
        photos,
        "Dlib cnn + landmarks",
        detect_face_and_landmarks_dlib,
        dnnFaceDetector,
        save_false_findings)
    run_detection(
        photos,
        "face recognition using hog",
        detect_face_face_recognition,
        "hog",
        save_false_findings)
    run_detection(
        photos,
        "face recognition using cnn",
        detect_face_face_recognition,
        "cnn",
        save_false_findings)


if __name__ == "__main__":
    main()
