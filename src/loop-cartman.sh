#!/bin/bash
source ../venv-tf15/Scripts/activate
while :
do
  echo "Looping a GPU (2x40) FredLiu-774 rest server. Press [CTRL+C] to stop."
  python flasky.py --http_host=0.0.0.0 --http_port=1303 --model_name=774M_southpark_convo --sample_size=2 --length=40 --gpu_phy=0
  sleep 2
done

# Note, the 2 processes we're using lately are:
#  --http_host=0.0.0.0 --http_port=1303 --model_name=774M_southpark_convo --length=40 --sample_size=2 --gpu_phy=0
#  --http_host=0.0.0.0 --http_port=1304 --model_name=774M_southpark_convo --length=4 --sample_size=2 --gpu_phy=1 --gpu_mem=0
