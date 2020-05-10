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
time [INFO] OpenCV Haar                found faces on 10, out of 13 pictures in   275 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 76% and mistake rate: 00%
time [INFO] OpenCV Dnn Caffe           found faces on 12, out of 13 pictures in   337 milliseconds. On  1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 92% and mistake rate: 07%
time [INFO] OpenCV Dnn Tf              found faces on 12, out of 13 pictures in   334 milliseconds. On  2 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 92% and mistake rate: 15%
time [INFO] Dlib Hog                   found faces on  9, out of 13 pictures in   201 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 69% and mistake rate: 00%
time [INFO] Dlib hog + landmarks       found faces on  9, out of 13 pictures in   202 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 69% and mistake rate: 00%
time [INFO] Dlib cnn                   found faces on 11, out of 13 pictures in   737 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 84% and mistake rate: 00%
time [INFO] Dlib cnn + landmarks       found faces on 11, out of 13 pictures in   112 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 84% and mistake rate: 00%
time [INFO] face recognition using hog found faces on  9, out of 13 pictures in   353 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 69% and mistake rate: 00%
time [INFO] face recognition using cnn found faces on 13, out of 13 pictures in   198 milliseconds. On  0 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 100% and mistake rate: 00%
</pre></code><br>

On sample, randomly scrapped from internet photos containing exactly one person on the picture
the following output was produced:

<code><pre>time [INFO] Total amount of photos: 1093
time INFO] OpenCV Haar                found faces on  996, out of 1093 pictures in 22816 milliseconds. On  171 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 91% and mistake rate: 15%
time [INFO] OpenCV Dnn Caffe           found faces on 1061, out of 1093 pictures in 28911 milliseconds. On   83 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 97% and mistake rate: 07%
time [INFO] OpenCV Dnn Tf              found faces on 1059, out of 1093 pictures in 27478 milliseconds. On   83 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 96% and mistake rate: 07%
time [INFO] Dlib Hog                   found faces on  866, out of 1093 pictures in 18979 milliseconds. On    3 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 79% and mistake rate: 00%
time [INFO] Dlib hog + landmarks       found faces on  866, out of 1093 pictures in 19085 milliseconds. On    3 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 79% and mistake rate: 00%
time [INFO] Dlib cnn                   found faces on  860, out of 1093 pictures in 12023 milliseconds. On    1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] Dlib cnn + landmarks       found faces on  860, out of 1093 pictures in 11626 milliseconds. On    1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] face recognition using hog found faces on  969, out of 1093 pictures in 40513 milliseconds. On    3 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 88% and mistake rate: 00%
time [INFO] face recognition using cnn found faces on 1012, out of 1093 pictures in 23855 milliseconds. On    1 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 92% and mistake rate: 00%
</pre></code>
 
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
