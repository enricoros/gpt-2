## Starting the 2 inference processes

### Process 1
2x40 on *:1303, Gpu0
```shell script
python flasky.py --http_host=0.0.0.0 --http_port=1303 --model_name=774M_southpark_convo --length=40 --sample_size=2 --gpu_phy=0
```

### Process 2
2x4 on *:2304, Gpu1, Auto-growing memory
```shell script
python flasky.py --http_host=0.0.0.0 --http_port=1304 --model_name=774M_southpark_convo --length=4 --sample_size=2 --gpu_phy=1 --gpu_mem=0
```
