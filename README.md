# Equitable and Network-Aware Decarbonization

This repository contains code and sample data for our eEnergy 2023 paper: Equitable Network-Aware Decarbonization of Residential Heating at City Scale

## Description of Sample Dataset

Example gas usage data for a single home (one gas meter) is provided in the file `101.csv`, inside the *gas_data* folder.

Example electric data for a single distribution transformer is provided in the file `123A.csv`, inside the *transformer_data* folder.

## Running the Experiments

To configure 

To compute the costs and carbon reduction achieved by converting different neighborhoods, run `python X.py`.

You can run the optimization by running the file `run_opt.py`

## Requirements

Python 3 is required.

The file *requirements.txt* lists the required packages.

The packages can be installed by running

`pip install -r requirements.txt`

## Publications

Please cite our paper if you use this code in academic work.

The BibTeX citation is given below.

```
@inproceedings{lechowicz2023,
  title = {Equitable Network-Aware Decarbonization of Residential Heating at City Scale},
  author = {Lechowicz, Adam and Bashir, Noman and Wamburu, John and Hajiesmaili, Mohammad and Shenoy, Prashant},
  booktitle = {Proceedings of the Fourteenth ACM International Conference on Future Energy Systems},
  year = {2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  series = {e-Energy '23}
}
```
