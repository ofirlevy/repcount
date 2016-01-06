introduction
----------------------------------------------------

This repository contains an implementation of Live Repetition counting method presented in ICCV2015 paper by Ofir Levy and Lior Wolf (Tel Aviv University).  
This method detects and live counts any type of repetative motion. Please refer the paper for more details

prerequisites
----------------------------------------------------

1. python 2.7 (might work also with python 3.x though we haven't tested it)
2. theano
3. python packages: cPickle, gzip, numpy, scipy, cv2
4. optional - Matlab if you want to retrain the CNN

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


running segmented benchmark
----------------------------------------------------

The 100 segmented videos are located in $ROOT/data/YT_seg folder.  
To run the system on this benchmark go to $ROOT/test_seg_benchmark.  
To run using online entropy
> python rep_test_benchmark.py --online

To run using offline entropy
> python rep_test_benchmark.py --offline

The offline entropy script will also present the stats of the median and best stride configurations.
See the paper for details regading online and offline entropy configuration.


create a synthetic training data set and train a classifier
----------------------------------------------------

1. Create synthetic data using our Matlab script.  
Go to $ROOT/syn_data and run cData_run.m  
This will create .mat files for train and validation sets under $ROOT/out/mat folder.
2. To convert the mat files to hdf files go to go to $ROOT/syn_data and run:  
  python rep_pickle.py  
This will create the require h5 files under $ROOT/out/h5 folder.
3. To train the network go to  $ROOT/trainNet folder and run:  
python rep_train_main.py  
A snapshot (weights file) will be stored every epoch under $ROOT/trainNet/weights folder.  
You can peak a weight file and replace it with the exisiting in the folders above.

