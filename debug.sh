#!/bin/sh

name="$1"
device="$2"
loglevel="20"

options="
    --name $name
    --loglevel $loglevel
    --unobserved
    "

config="configs/$name.yaml"

run="
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$device
    python run.py $options with $config ${@:3}
    "

printf "$run \n\n"
eval $run