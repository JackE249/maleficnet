#!/bin/bash

for p in ./payload/*
do
	echo "injecting ${p}"
	python3.9 maleficnet.py --epoch 10 --model densenet --payload "${p:10}" --gamma 0.0009 --dataset cifar10 --num_classes 10 --dim 32
done

