#!/bin/bash
source ../venv-tf15/Scripts/activate
while :
do
  echo "Press [CTRL+C] to stop.."
  python flasky.py --http_host=0.0.0.0 --http_port=1303 --model_name=345M-Cartman-Fredliu1 --sample_size=4 --length=60
done
