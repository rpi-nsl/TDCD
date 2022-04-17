#!/bin/bash -x
# activate conda environment first if necessary --> https://hpc-discourse.usc.edu/t/conda-activate-error/252
# eval "$(conda shell.bash hook)"
# conda activate flearning

#Cifar10 50 clients
python cifar_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 60000 --Q 1 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 6000 --Q 10 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 3000 --Q 20 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 2000 --Q 30 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 

#Cifar10 100 clients
python cifar_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 60000 --Q 1 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 6000 --Q 10 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 3000 --Q 20 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 2000 --Q 30 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
