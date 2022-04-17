# Tiered Decentralized Coordinate-Descent (TDCD)

### Dependencies
One can install our environment with Anaconda:
    conda env create -f flearn.yml 

### Dataset
ModelNet40 12-view PNG dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/0B4v2jR3WsindMUE3N2xiLVpyLW8/view).
The code expects the dataset to be present in the current working directory in a folder named "view".

Prerequisite for Mimic3: 
Follow the instructions in [mimic3-benchmark](https://github.com/YerevaNN/mimic3-benchmarks) to download and pre-sanitize the datasets. After the extraction and sanitization is done, put the raw train, test and validation files in  `~/data/mimi3lstm` directory. 

### Run the batch scripts for running batch experiments
To run experiments using the batch scripts you can use the shell scripts we have provided.

For Cifar10 experiments, run:
```bash
./sbatch_cifar.sh
```
For Modelnet40 experiments, run:
```bash
./sbatch_modelnet.sh
```
For Mimic3 experiments, run:
```bash
./sbatch_mimic3.sh
```

#### Run the Cifar10 experiments individually using the python scripts

```bash
python cifar_journal.py [--seed [Seed to use]] [--hubs [Number of hubs/silos]]
                  [--clients [Number of clients in each silo]] 
                  [--gepochs [Number of global iterations]] 
                  [--Q [Number of local iterations per global iteration]]
                  [--batchsize [Overall number of samples selected in each iteration across all clients in each silo]] 
                  [--lr [Learning Rate of SGD]] 
                  [--momentum [Momentum in SGD (not used)]] 
                  [--evalafter [Evaluate after number of local iterations]]
```

#### Run the Modelnet40 experiments individually using the python scripts

```bash
python modelnet_journal.py [--seed [Seed to use]] [--hubs [Number of hubs/silos]]
                  [--clients [Number of clients in each silo]] 
                  [--gepochs [Number of global iterations]] 
                  [--Q [Number of local iterations per global iteration]]
                  [--batchsize [Overall number of samples selected in each iteration across all clients in each silo]] 
                  [--lr [Learning Rate of SGD]] 
                  [--momentum [Momentum in SGD (not used)]] 
                  [--evalafter [Evaluate after number of local iterations]]
                  [--modelnet_type [Determine the type of Modelnet used. Possible values ModelNet10 or ModelNet40.]]

```

#### Run the Mimic3 experiments individually using the python scripts

```bash
python mimic3_journal.py [--seed [Seed to use]] [--hubs [Number of hubs/silos]]
                  [--clients [Number of clients in each silo]] 
                  [--gepochs [Number of global iterations]] 
                  [--Q [Number of local iterations per global iteration]]
                  [--batchsize [Overall number of samples selected in each iteration across all clients in each silo]] 
                  [--lr [Learning Rate of SGD]] 
                  [--momentum [Momentum in SGD (not used)]] 
                  [--evalafter [Evaluate after number of local iterations]]
```

#### Sample plot generating scripts are provided in `~/plot_utils/
