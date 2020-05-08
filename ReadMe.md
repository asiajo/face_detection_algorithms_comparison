# Face detection algorithms comparison

Created for comparing time and accuracy of face detection algorithms.<br>
Based on the code from LearnOpenCV.com.

## Sample input

Set of 13 pictures is provided. Path to this set is hardcoded in the program.
Input pictures are expected to contain exactly one person on the photo.
Based on this assumption the algorithms are compared.

## Sample output.

On those sample pictures following output was generated (with GPU support):
<br><code><pre>time [INFO] Total amount of photos: 13
time [INFO] OpenCV Haar                found faces on 10, out of 13 pictures in   291 milliseconds. On  2 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 76% and mistake rate: 15%
time [INFO] OpenCV Dnn Caffe           found faces on 12, out of 13 pictures in   344 milliseconds. On  1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 92% and mistake rate: 07%
time [INFO] OpenCV Dnn Tf              found faces on 12, out of 13 pictures in   351 milliseconds. On  1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 92% and mistake rate: 07%
time [INFO] Dlib Hog                   found faces on  9, out of 13 pictures in   205 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 69% and mistake rate: 00%
time [INFO] Dlib cnn                   found faces on 11, out of 13 pictures in   810 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 84% and mistake rate: 00%
time [INFO] face recognition using hog found faces on 10, out of 13 pictures in   476 milliseconds. On  1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 76% and mistake rate: 07%
time [INFO] face recognition using cnn found faces on 13, out of 13 pictures in   288 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 100% and mistake rate: 00%
</pre></code><br>

On sample, randomly scrapped from internet photos containing exactly one person on the picture
the following output was produced:

<code><pre>time [INFO] Total amount of photos: 1127
time [INFO] OpenCV Haar                found faces on 1027, out of 1127 pictures in 24360 milliseconds. On  162 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 91% and mistake rate: 14%
time [INFO] OpenCV Dnn Caffe           found faces on 1086, out of 1127 pictures in 29609 milliseconds. On   88 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 96% and mistake rate: 07%
time [INFO] OpenCV Dnn Tf              found faces on 1087, out of 1127 pictures in 28599 milliseconds. On   87 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 96% and mistake rate: 07%
time [INFO] Dlib Hog                   found faces on  880, out of 1127 pictures in 19837 milliseconds. On    6 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] Dlib cnn                   found faces on  876, out of 1127 pictures in 12952 milliseconds. On    5 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 77% and mistake rate: 00%
time [INFO] face recognition using hog found faces on 1012, out of 1127 pictures in 54771 milliseconds. On    9 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 89% and mistake rate: 00%
time [INFO] face recognition using cnn found faces on 1053, out of 1127 pictures in 35579 milliseconds. On   11 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 93% and mistake rate: 00%
</pre></code>
 
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
