#!/usr/bin/env bash

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_DIR=$CURRENT_DIR/../data

if [ ! -f $DATA_DIR/CamVid.zip ]; then
    echo "Downloading Camvid dataset using git lfs"
    git lfs pull
fi

unzip $DATA_DIR/CamVid.zip -d $DATA_DIR/
