#!/bin/sh

name="$1"
loglevel="20"
mongo_db="drunk:27017:$name.runs"

options="
    --name $name
    --loglevel $loglevel
    --mongo_db $mongo_db
    "

config="configs/$name.yaml"

run="python run.py $options with $config ${@:2}"

printf "$run"
eval $run