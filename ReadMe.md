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

<code><pre>time [INFO] Total amount of photos: 1111
time [INFO] OpenCV Haar                found faces on 1010, out of 1111 pictures in 24524 milliseconds. On  175 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 90% and mistake rate: 15%
time [INFO] OpenCV Dnn Caffe           found faces on 1075, out of 1111 pictures in 28810 milliseconds. On   84 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 96% and mistake rate: 07%
time [INFO] OpenCV Dnn Tf              found faces on 1073, out of 1111 pictures in 27900 milliseconds. On   84 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 96% and mistake rate: 07%
time [INFO] Dlib Hog                   found faces on  874, out of 1111 pictures in 19319 milliseconds. On    4 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] Dlib hog + landmarks       found faces on  874, out of 1111 pictures in 19313 milliseconds. On    4 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] Dlib cnn                   found faces on  867, out of 1111 pictures in 12636 milliseconds. On    3 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] Dlib cnn + landmarks       found faces on  867, out of 1111 pictures in 11923 milliseconds. On    3 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 78% and mistake rate: 00%
time [INFO] face recognition using hog found faces on  979, out of 1111 pictures in 41325 milliseconds. On    5 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 88% and mistake rate: 00%
time [INFO] face recognition using cnn found faces on 1021, out of 1111 pictures in 24422 milliseconds. On    3 pictures it made a mistake and found more than one face. Achieved accuracy of correct findings: 91% and mistake rate: 00%

Process finished with exit code 0
</pre></code>
 
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
