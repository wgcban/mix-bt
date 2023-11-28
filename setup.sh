#!/bin/bash

mkdir ssl-aug
mkdir Barlow-Twins-HSIC
mkdir /data/wbandar1/projects/ssl-aug-artifacts/results
git clone https://github.com/wgcban/ssl-aug.git Barlow-Twins-HSIC

cd Barlow-Twins-HSIC
conda env create -f environment.yml
conda activate ssl-aug


