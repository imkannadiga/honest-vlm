#!/bin/bash
accelerate launch \
    --num_processes=4 \
    --num_machines=2 \
    --machine_rank=1 \
    --main_process_ip="10.69.180.212" \
    --main_process_port=29500 \
    train.py
