# Face detection algorithms comparison

Created for comparing time and accuracy of face detection algorithms.

## Sample input

Set of 13 pictures is provided. Path to this set is hardcoded in the program.
Input pictures are expected to contain exactly one person on the photo.
Based on this assumption the algorithms are compared.

## Sample output.

On those sample pictures following output was generated (with GPU support):
<br><code><pre>Total amount of photos with exactly one face on the image: 13. Smaller edge length of the image fed to the network: 300 px.
| Network:                         | total images:     | found face on:  | in time:        | more than one face on: | accuracy:       | mistake rate:   |
| OpenCV Haar + landmarks          |              13 |              10 |          293 ms |                      1 |            76 % |             7 % |
| OpenCV Dnn Caffe just rectangles |              13 |              12 |          355 ms |                      2 |            92 % |            15 % |
| OpenCV Dnn Tf    just rectangles |              13 |              12 |          347 ms |                      2 |            92 % |            15 % |
| Dlib hog + landmarks             |              13 |               9 |          203 ms |                      0 |            69 % |             0 % |
| Dlib cnn + landmarks             |              13 |              11 |          745 ms |                      0 |            84 % |             0 % |
| face_recognition using hog       |              13 |               9 |          778 ms |                      0 |            69 % |             0 % |
| face_recognition using cnn       |              13 |              13 |          422 ms |                      0 |           100 % |             0 % |</pre></code><br>

On sample, randomly scrapped from internet photos containing exactly one person on the picture
the following output was produced:

<code><pre>Total amount of photos with exactly one face on the image: 1073. Smaller edge length of the image fed to the network: 300 px.
| Network:                         | total images:     | found face on:  | in time:        | more than one face on: | accuracy:       | mistake rate:   |
| OpenCV Haar + landmarks          |            1073 |             984 |        27908 ms |                    189 |            91 % |            17 % |
| OpenCV Dnn Caffe just rectangles |            1073 |            1042 |        28328 ms |                      2 |            97 % |             0 % |
| OpenCV Dnn Tf    just rectangles |            1073 |            1041 |        27704 ms |                      2 |            97 % |             0 % |
| Dlib hog + landmarks             |            1073 |             901 |        22433 ms |                      4 |            83 % |             0 % |
| Dlib cnn + landmarks             |            1073 |             910 |        13533 ms |                      0 |            84 % |             0 % |
| face_recognition using hog       |            1073 |            1000 |        88142 ms |                      5 |            93 % |             0 % |
| face_recognition using cnn       |            1073 |            1049 |        50676 ms |                      3 |            97 % |             0 % |</pre></code>

<code><pre>Total amount of photos with exactly one face on the image: 1073. Smaller edge length of the image fed to the network: 200 px.
| Network:                         | total images:     | found face on:  | in time:        | more than one face on: | accuracy:       | mistake rate:   |
| OpenCV Haar + landmarks          |            1073 |             959 |        16157 ms |                    103 |            89 % |             9 % |
| OpenCV Dnn Caffe just rectangles |            1073 |            1041 |        28292 ms |                      2 |            97 % |             0 % |
| OpenCV Dnn Tf    just rectangles |            1073 |            1039 |        27678 ms |                      2 |            96 % |             0 % |
| Dlib hog + landmarks             |            1073 |             728 |        10887 ms |                      2 |            67 % |             0 % |
| Dlib cnn + landmarks             |            1073 |             664 |         7308 ms |                      0 |            61 % |             0 % |
| face_recognition using hog       |            1073 |             951 |        39880 ms |                      3 |            88 % |             0 % |
| face_recognition using cnn       |            1073 |             994 |        23651 ms |                      0 |            92 % |             0 % |</pre></code>
 
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
