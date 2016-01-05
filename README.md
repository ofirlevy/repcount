introduction
----------------------------------------------------

This repository contains an implementation of Live Repetition counting method presented in ICCV2015 paper by Ofir Levy and Lior Wolf (Tel Aviv University).  
This method detects and live counts any type of repetative motion. Please refer the paper for more details

prerequisites
----------------------------------------------------

1. python 2.7 (might work also with python 3.x though we haven't tested it)
2. theano
3. python packages: cPickle, gzip, numpy, scipy, cv2

note: we refer below to $ROOT as the root folder of this repository.

counting live from camera
----------------------------------------------------

This script operates our live counting system using a webcam as input.  
1. make sure you have webcam connected  
2. go to $ROOT/live_count folder and run:
> python live_rep.py 

Alternatively, You can stream from file using:  
> python live_rep.py -i "file_name"

You can try as an input our captured long live video, located at $ROOT/data/cam. i.e: 
> python live_rep.py -i ../data/cam/live.avi

The output video will be stored at $ROOT/out folder

running YTIO benchmark
----------------------------------------------------

The 25 YTIO videos are located in $ROOT/data/YTIO folder.  
To run the system on this benchmark go to $ROOT/live_count folder and run:
> python live_rep_YTIO.py 

The output videos will be stored at $ROOT/out folder


running segmented benchmark with online entropy
----------------------------------------------------

The 25 YTIO videos are located in $ROOT/data/YT_seg folder.  
To run the system on this benchmark go to $ROOT/test_seg_benchmark.  
To run using online entropy
> python live_rep_YTIO.py --online


See the paper for details regading online and offline entropy configuration





create a synthetic training data set and classifier
----------------------------------------------------

