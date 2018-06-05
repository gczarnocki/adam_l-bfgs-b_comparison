#!/bin/bash

conda create -n coco python=2.7 anaconda
. activate coco

pip install -r requirements.txt