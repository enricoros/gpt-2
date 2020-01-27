#!/bin/bash
source ../venv-tf15/Scripts/activate
while :
do
  echo "Looping a GPU (2x40) FredLiu-774 rest server. Press [CTRL+C] to stop."
  python flasky.py --http_host=0.0.0.0 --http_port=1303 --model_name=774M_southpark_convo --sample_size=2 --length=40
  sleep 2
done
