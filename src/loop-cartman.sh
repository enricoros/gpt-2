#!/bin/bash
while :
do
  echo "Press [CTRL+C] to stop.."
  python flasky.py --http_host=0.0.0.0 --http_port=1303 --model_name=124M-Cartman-Fredliu0 --sample_size=4 --length=60
done
