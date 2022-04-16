#!/bin/bash -x
# activate conda environment first --> https://hpc-discourse.usc.edu/t/conda-activate-error/252
eval "$(conda shell.bash hook)"
conda activate flearning

# ModelNet40 Clients=10
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 10 --gepochs 60000 --Q 1 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 10 --gepochs 6000 --Q 10 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 10 --gepochs 3000 --Q 20 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 10 --gepochs 2000 --Q 30 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 

# ModelNet40 Clients=20
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 20 --gepochs 60000 --Q 1 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 20 --gepochs 6000 --Q 10 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 20 --gepochs 3000 --Q 20 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
python modelnet_restructure_journal_tim.py --seed 2021 --hubs 12 --clients 20 --gepochs 2000 --Q 30 --batchsize 320 --lr 0.0005 --momentum 0.0 --evalafter 20 --modelnet_type 40 
