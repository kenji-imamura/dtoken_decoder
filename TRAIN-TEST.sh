#!/bin/bash
#
# An example of the sequence of whole training and test.
#

bash TRAIN0.sh
bash TRAIN1.sh 
bash TRAIN2.bash -g 0,1,2,3 -s en -t de
bash TEST.bash -g 0 -c newstest2013 ./newstest2013
