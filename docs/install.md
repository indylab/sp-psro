# Installation
(Tested on Ubuntu 20.04)

### Overview

1. Clone the repo.
2. Set up a Conda env.
3. Install python modules (including bundled dependencies).


### Clone repo with git submodules
```shell script
git clone --recursive git@github.com:indylab/sp-psro.git
cd grl
```
If you've already cloned this repo but not the [submodules](/dependencies), you can clone them with:
```shell script
git submodule update --init --recursive
```

### Set up Conda environment
After installing [Anaconda](https://docs.anaconda.com/anaconda/install/), enter the repo directory and create the new environment:
```shell script
conda env create -f environment.yml
conda activate sp_psro
```

### Install Python modules

#### 1. DeepMind OpenSpiel (included dependency)
DeepMind's [OpenSpiel](https://github.com/deepmind/open_spiel) is used for poker game logic as well as tabular game utilities.
We include a slightly modified [fork](https://github.com/indylab/open_spiel/tree/spsro-release) as a dependency.
```shell script
# Starting from the repo root
cd dependencies/open_spiel
./install.sh
OPEN_SPIEL_BUILD_WITH_ACPC=OFF OPEN_SPIEL_BUILD_WITH_HANABI=OFF pip install -e . # This will start a compilation process. May take a few minutes.
cd ../..
```

#### 2. The GRL Package (main package).
```shell script
# Starting from the repo root
pip install -e .
```


### Next Steps

- [Running APSRO Experiments](/docs/apsro.md)
- [Running SP-PSRO Experiments](/docs/sp-psro.md)