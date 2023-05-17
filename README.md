ACR Attacks
=====================

This repository provides the artifacts for the paper `Listening between the Bits: Privacy Leaks in Audio Fingerprints`.


Quickstart Guide
----------------

First, install the dependencies as follows.

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Afterwards, to reproduce the results from our paper please follow the following steps.

1. Contact the authors to receive the dataset. (**Our paper is currently under submission. Authors contacts will be disclosed after publication.**) 
2. Copy the file `acr_privacy_attacks_datasets.tar.gz` next to the `Makefile`.
3. Export the variable `FINGERPRINT_ROOT` and `EXPERIMENT_ROOT` to specify the directories for input and output data, e.g. with
```bash
export FINGERPRINT_ROOT=$(pwd)/data
export EXPERIMENT_ROOT=$(pwd)/results
```
4. Run `make prepare-datasets`.
5. Run `make eval-all`.

Project Organization
--------------------

	├── Makefile                       <- Use `make fingerprints` or `make eval-attacks`
	├── README.md                      <- Top-level README to inform about general usage.
	├── requirements.txt               <- The requirements file for reproducing the analysis environment.
	├── src
	│   ├── attacking                  <- Scripts to run attacks on ACR solutions
	│   │   ├── attacks                <- Attack models
	│   │   ├── common                 <- Configuration file and various helper functions
	│   │   ├── evaluation             <- Scripts to start the evaluation
	│   │   ├── features               <- Feature extractors of all three considered ACR solutions
	│   │   ├── labelencoders          <- Label encoding for speaker experiments
	│   │   ├── README.md              <- README describing how to re-run evaluation of this project
	│   │   ├── requirements.txt
	│   │   └── run_all.sh
	│   ├── fingerprinting             <- Scripts to turn the preprocessed data into fingerprints.
	│   │   ├── acrcloud               <- Python and Frida Scripts to extract fingerprints from ACRCloud.
	│   │   ├── audio                  <- Module that is being used for audio formatting.
	│   │   ├── common                 <- Additional helper library to handle the fingerprint files.
	│   │   ├── make_fingerprints.py   <- Executes all the other python scripts to generate the fingerprints.
	│   │   ├── README.md
	│   │   ├── sonnleitner            <- Contains all the python scripts to generate sonnleitner fingerprints.
	│   │   └── zapr                   <- Includes scripts to retrieve fingerprints from com.winit.starnews.hin.
