#!/bin/bash

for i in {1..20}
do
	python3.9 no_inject.py --epoch 10 --model densenet --payload ssssss --gamma 0.0009 --dataset cifar10 --num_classes 10 --dim 32 --random_seed "$i" --only_pretrained
done

for i in {21..40}
do
	python3.9 no_inject.py --epoch 10 --model densenet --payload ssssss --gamma 0.0009 --dataset cifar10 --num_classes 10 --dim 32 --random_seed "$i"
done