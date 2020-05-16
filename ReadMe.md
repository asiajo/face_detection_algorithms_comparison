# Face detection algorithms comparison

Created for comparing time and accuracy of face detection algorithms.<br>
Based on the code from LearnOpenCV.com.

## Sample input

Set of 13 pictures is provided. Path to this set is hardcoded in the program.
Input pictures are expected to contain exactly one person on the photo.
Based on this assumption the algorithms are compared.

## Sample output.

On those sample pictures following output was generated (with GPU support):
<br><code><pre>Total amount of photos with exactly one face on the image: 13
| Network:                   | total imgs:     | found face on:  | in time:        | more than one face on:     | accuracy:       | mistake rate:   |
| OpenCV Haar                |              13 |              10 |          278 ms |                          0 |            76 % |             0 % |
| OpenCV Dnn Caffe           |              13 |              12 |          348 ms |                          1 |            92 % |             7 % |
| OpenCV Dnn Tf              |              13 |              12 |          339 ms |                          2 |            92 % |            15 % |
| Dlib Hog                   |              13 |               9 |          202 ms |                          0 |            69 % |             0 % |
| Dlib hog + landmarks       |              13 |               9 |          202 ms |                          0 |            69 % |             0 % |
| Dlib cnn                   |              13 |              11 |          752 ms |                          0 |            84 % |             0 % |
| Dlib cnn + landmarks       |              13 |              11 |          109 ms |                          0 |            84 % |             0 % |
| face recognition using hog |              13 |               9 |          355 ms |                          0 |            69 % |             0 % |
| face recognition using cnn |              13 |              13 |          205 ms |                          0 |           100 % |             0 % |</pre></code><br>

On sample, randomly scrapped from internet photos containing exactly one person on the picture
the following output was produced:

<code><pre>Total amount of photos with exactly one face on the image: 1073
| Network:                   | total imgs:     | found face on:  | in time:        | more than one face on:     | accuracy:       | mistake rate:   |
| OpenCV Haar                |            1073 |             979 |        22394 ms |                        167 |            91 % |            15 % |
| OpenCV Dnn Caffe           |            1073 |            1041 |        27092 ms |                          2 |            97 % |             0 % |
| OpenCV Dnn Tf              |            1073 |            1039 |        26437 ms |                          2 |            96 % |             0 % |
| Dlib Hog                   |            1073 |             849 |        18611 ms |                          3 |            79 % |             0 % |
| Dlib hog + landmarks       |            1073 |             849 |        18643 ms |                          3 |            79 % |             0 % |
| Dlib cnn                   |            1073 |             846 |        11848 ms |                          0 |            78 % |             0 % |
| Dlib cnn + landmarks       |            1073 |             846 |        11409 ms |                          0 |            78 % |             0 % |
| face recognition using hog |            1073 |             951 |        39836 ms |                          3 |            88 % |             0 % |
| face recognition using cnn |            1073 |             994 |        23506 ms |                          0 |            92 % |             0 % |</pre></code>
 
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
