# Equitable and Network-Aware Decarbonization

This repository contains code and sample data for our eEnergy 2023 paper: Equitable Network-Aware Decarbonization of Residential Heating at City Scale

## Description of Sample Dataset

Example gas usage data for a single home (one gas meter) is provided in the file `101.csv`, inside the *gas_data* folder.

Example electric data for a single distribution transformer is provided in the file `123A.csv`, inside the *transformer_data* folder.

Example census data for the above gas meter and transformer is provided in the *census_data* folder.

The main mapping example in 'MainMappingExample.csv' contains an example (for one household) of the mapping between electric meter ID, transformer ID, gas meter ID, and the location of the household (should be within the experimental municipality).

## Configuration

Because the code constructs a simulated natural gas network directly from OpenStreetMap data using the [OSMnx](https://github.com/gboeing/osmnx) package, a data set for any municipality must contain sensititve information (locations of gas consumers and their energy usage).  In this repository, we provide sample data which provide the correct format for each category of data.

Before running experiments, a few configuration steps are required:

1. Set the municipality location in 'gasNet.py' and 'naive.py'.  This defines the OpenStreetMap region for which the simulated graph will be generated.
2. In '.env', set the environment variables for the location of data folders, the OSM ID for the node closest to the municipality's [gate station](https://www.sciencedirect.com/topics/engineering/gate-station), and the median cost for installing a heat pump system in a house.
3. For equity analysis, given a complete main mapping and census data, running the 'incomeMapping.ipynb' notebook generates a Python dictionary which classifies the households in the municipality according to low, medium, and high income tracts.

## Running the Experiments

To compute the costs and carbon reduction achieved by converting different neighborhoods, run `python calculateValues.py`.

To solve the optimization for a given budget, run `python solver.py results.csv`

To solve the equity-aware optimization for a given budget, run `python equitySolver.py results.csv`

## Requirements

Python 3, conda, and Jupyter are required.

The necessary packages can be installed by following the [OSMnx installation guide](https://osmnx.readthedocs.io/en/stable/), or equivalently by running the two commands below in a conda environment.

`conda config --prepend channels conda-forge\n conda create -n ox --strict-channel-priority osmnx jupyterlab pandas`

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
