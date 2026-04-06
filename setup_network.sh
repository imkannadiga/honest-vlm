#!/bin/bash
# Run 'ip a' or 'ifconfig' in your terminal to find your actual network interface name.
# It is usually eth0, eno1, or enp3s0.
export NCCL_SOCKET_IFNAME=eno1 
export NCCL_DEBUG=INFO

echo "NCCL network interface forced to: $NCCL_SOCKET_IFNAME"