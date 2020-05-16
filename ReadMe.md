# Face detection algorithms comparison

Created for comparing time and accuracy of face detection algorithms.

## Sample input

Set of 13 pictures is provided. Path to this set is hardcoded in the program.
Input pictures are expected to contain exactly one person on the photo.
Based on this assumption the algorithms are compared.

## Sample output.

On those sample pictures following output was generated (with GPU support):
<br><code><pre>Total amount of photos with exactly one face on the image: 13. Smaller edge length of the image fed to the network: 200 px.
| Network:                   | total images:   | found face on:  | in time:        | more than one face on: | accuracy:       | mistake rate:   |
| OpenCV Haar                |              13 |              10 |          215 ms |                      0 |            76 % |             0 % |
| OpenCV Dnn Caffe           |              13 |              12 |          362 ms |                      1 |            92 % |             7 % |
| OpenCV Dnn Tf              |              13 |              12 |          356 ms |                      2 |            92 % |            15 % |
| Dlib hog                   |              13 |               9 |          104 ms |                      0 |            69 % |             0 % |
| Dlib cnn                   |              13 |              11 |          713 ms |                      0 |            84 % |             0 % |
| face_recognition using hog |              13 |               9 |          357 ms |                      0 |            69 % |             0 % |
| face_recognition using cnn |              13 |              13 |          196 ms |                      0 |           100 % |             0 % |</pre></code><br>

On sample, randomly scrapped from internet photos containing exactly one person on the picture
the following output was produced:

<code><pre>Total amount of photos with exactly one face on the image: 1072. Smaller edge length of the image fed to the network: 300 px.
| Network:                   | total images:   | found face on:  | in time:        | more than one face on: | accuracy:       | mistake rate:   |
| OpenCV Haar                |            1072 |             983 |        27608 ms |                    188 |            91 % |            17 % |
| OpenCV Dnn Caffe           |            1072 |            1041 |        28436 ms |                      1 |            97 % |             0 % |
| OpenCV Dnn Tf              |            1072 |            1040 |        27886 ms |                      1 |            97 % |             0 % |
| Dlib hog                   |            1072 |             901 |        22482 ms |                      4 |            84 % |             0 % |
| Dlib cnn                   |            1072 |             910 |        13550 ms |                      0 |            84 % |             0 % |
| face_recognition using hog |            1072 |             999 |        88715 ms |                      5 |            93 % |             0 % |
| face_recognition using cnn |            1072 |            1048 |        51060 ms |                      2 |            97 % |             0 % |</pre></code>

<code><pre>Total amount of photos with exactly one face on the image: 1072. Smaller edge length of the image fed to the network: 200 px.
| Network:                   | total images:   | found face on:  | in time:        | more than one face on: | accuracy:       | mistake rate:   |
| OpenCV Haar                |            1072 |             958 |        16064 ms |                    102 |            89 % |             9 % |
| OpenCV Dnn Caffe           |            1072 |            1040 |        28284 ms |                      1 |            97 % |             0 % |
| OpenCV Dnn Tf              |            1072 |            1038 |        27718 ms |                      1 |            96 % |             0 % |
| Dlib hog                   |            1072 |             728 |        10927 ms |                      2 |            67 % |             0 % |
| Dlib cnn                   |            1072 |             664 |         7302 ms |                      0 |            61 % |             0 % |
| face_recognition using hog |            1072 |             950 |        40001 ms |                      3 |            88 % |             0 % |
| face_recognition using cnn |            1072 |             993 |        23582 ms |                      0 |            92 % |             0 % |</pre></code>
 
It also generates folder with wrongly classified pictures. If too many faces were detected on the photo - it draws rectangles around found faces.

## Notes:

Developed and tested on Ubuntu 18 with CUDA 10.1.
