#!/bin/sh

name="$1"
device="$2"
loglevel="20"
mongo_db="drunk:27017:$name.runs"

options="
    --name $name
    --loglevel $loglevel
    --mongo_db $mongo_db
    "

config="configs/$name.yaml"

run="
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$device
    python run.py $options with $config ${@:3}
    "

printf "$run \n"
eval $run    