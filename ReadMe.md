# Face detection algorithms comparison

Created for comparing time and accuracy of face detection algorithms.<br>
Based on the code from LearnOpenCV.com.

## Sample input

Set of 13 pictures is provided. Path to this set is hardcoded in the program.
Input pictures are expected to contain exactly one person on the photo.
Based on this assumption the algorithms are compared.

## Sample output.

On those sample pictures following output was generated (with GPU support):
<br><br>
time [INFO] Total amount of photos: 13<br>
time [INFO] OpenCV Haar                found faces on 10, out of 13 pictures in 433 milliseconds. On 1 pictures it made a mistake and found more than one face.<br>
time [INFO] OpenCV Dnn Caffe           found faces on 12, out of 13 pictures in 532 milliseconds. On 2 pictures it made a mistake and found more than one face.<br>
time [INFO] OpenCV Dnn Tf              found faces on 12, out of 13 pictures in 431 milliseconds. On 2 pictures it made a mistake and found more than one face.<br>
time [INFO] Dlib Hog                   found faces on 9, out of 13 pictures in 313 milliseconds. On 0 pictures it made a mistake and found more than one face.<br>
time [INFO] Dlib cnn                   found faces on 11, out of 13 pictures in 808 milliseconds. On 0 pictures it made a mistake and found more than one face.<br>
time [INFO] face recognition using hog found faces on 9, out of 13 pictures in 11795 milliseconds. On 0 pictures it made a mistake and found more than one face.
<br><br>
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
