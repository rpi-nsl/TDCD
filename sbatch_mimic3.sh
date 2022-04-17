#!/bin/bash -x
# activate conda environment first if necessary --> https://hpc-discourse.usc.edu/t/conda-activate-error/252
# eval "$(conda shell.bash hook)"
# conda activate flearning

python mimic3_journal.py --seed 2021 --hubs 4 --clients 20 --gepochs 24000 --Q 1 --batchsize 2000 --lr 0.1 --momentum 0.0 --evalafter 20
python mimic3_journal.py --seed 2021 --hubs 4 --clients 20 --gepochs 2400 --Q 10 --batchsize 2000 --lr 0.01 --momentum 0.0 --evalafter 20
python mimic3_journal.py --seed 2021 --hubs 4 --clients 20 --gepochs 1200 --Q 20 --batchsize 2000 --lr 0.01 --momentum 0.0 --evalafter 20
python mimic3_journal.py --seed 2021 --hubs 4 --clients 20 --gepochs 800 --Q 30 --batchsize 2000 --lr 0.01 --momentum 0.0 --evalafter 20

python mimic3_journal.py --seed 2021 --hubs 4 --clients 50 --gepochs 24000 --Q 1 --batchsize 2000 --lr 0.1 --momentum 0.0 --evalafter 20
python mimic3_journal.py --seed 2021 --hubs 4 --clients 50 --gepochs 2400 --Q 10 --batchsize 2000 --lr 0.01 --momentum 0.0 --evalafter 20
python mimic3_journal.py --seed 2021 --hubs 4 --clients 50 --gepochs 1200 --Q 20 --batchsize 2000 --lr 0.01 --momentum 0.0 --evalafter 20
python mimic3_journal.py --seed 2021 --hubs 4 --clients 50 --gepochs 800 --Q 30 --batchsize 2000 --lr 0.01 --momentum 0.0 --evalafter 20

python mimic3_journal.py --seed 2021 --hubs 16 --clients 50 --gepochs 2400 --Q 10 --batchsize 2000 --lr 0.1 --momentum 0.0 --evalafter 20