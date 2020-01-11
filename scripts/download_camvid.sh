#!/usr/bin/env bash

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_DIR=$CURRENT_DIR/../data

mkdir -p $DATA_DIR/camvid

if [ ! -f $DATA_DIR/CamSeq01.zip ]; then
    echo "Downloading Camvid dataset"
    wget http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip -P $DATA_DIR
fi

unzip $DATA_DIR/CamSeq01.zip -d $DATA_DIR/camvid
