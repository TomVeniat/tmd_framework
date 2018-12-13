#!/bin/sh

name="$1"
loglevel="20"

options="
    --name $name
    --loglevel $loglevel
    --unobserved
    "

config="configs/$name.yaml"

run="python run.py $options with $config ${@:2}"

printf "$run \n\n"
eval $run