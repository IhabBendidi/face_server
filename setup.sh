#!/bin/bash


apt-get update
sudo apt-get install python python-pip build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
pip3 install --upgrade setuptools
pip3 install -r requirements.txt
