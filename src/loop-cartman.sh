#!/bin/bash
source ../venv-tf15/Scripts/activate
while :
do
  echo "Press [CTRL+C] to stop.."
  python flasky.py --http_host=0.0.0.0 --http_port=1303 --model_name=774M_southpark_convo --sample_size=2 --length=40
done
