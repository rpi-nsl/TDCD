#!/bin/bash -x
# activate conda environment first --> https://hpc-discourse.usc.edu/t/conda-activate-error/252
eval "$(conda shell.bash hook)"
conda activate flearning

python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 60000 --Q 1 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 6000 --Q 10 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 3000 --Q 20 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 50 --gepochs 2000 --Q 30 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 

python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 60000 --Q 1 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 6000 --Q 10 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 3000 --Q 20 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
python cifar_restructure_journal.py --seed 2021 --hubs 2 --clients 100 --gepochs 2000 --Q 30 --batchsize 2000 --lr 0.0001 --momentum 0.0 --evalafter 20 
