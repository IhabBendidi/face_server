#!/bin/bash


apt-get update
apt-get install -y build-essential cmake pkg-config
apt-get install -y libx11-dev libatlas-base-dev
apt-get install -y libgtk-3-dev libboost-python-dev

apt-get install -y python3-dev python3-pip

python3 -m pip install -r requirements.txt
