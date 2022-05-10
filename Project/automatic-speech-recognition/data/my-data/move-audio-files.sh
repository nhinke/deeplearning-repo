#!/bin/bash

#
# to start recording studio:
#   $ cd ~/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/mimic-recording-studio
#   $ sudo docker-compose up
#

AUDIODIR=audio_files
NEWAUDIOPATH=~/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/my-data
ORIGAUDIOPATH=~/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/mimic-recording-studio/backend/audio_files/1f1e4444-404b-ee95-bbb8-37eb2223425c
PYTHONPATH=~/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/convert_txt_to_csv.py

# copy audio files to appropriate directory
([ -d $NEWAUDIOPATH/$AUDIODIR ] && rm -r $NEWAUDIOPATH/$AUDIODIR)
(cd $NEWAUDIOPATH && mkdir $AUDIODIR)
cp $ORIGAUDIOPATH/*.wav $NEWAUDIOPATH/$AUDIODIR

# run python script to create metadata .csv file
python3 $PYTHONPATH
