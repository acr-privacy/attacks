# Evaluation

Install dependencies with

```bash
python3.8 -m venv .acr-venv
source .acr-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set environment variables, e.g.:

```bash
export FINGERPRINT_ROOT=/data/acr/fingerprints            # Folder containing generated audio fingerprints
export EXPERIMENT_ROOT=/data/acr/experiments/speakers     # Result dir (e.g., for speaker experiments)
export PYTHONPATH=${PATH_TO_REPO}/acr-ccs-artifacts/src
```

Change to src directory and run experiments with
```bash
cd ${PROJECT_DIR}/acr-ccs-artifacts/src
./run_all.sh ${PROJECT_DIR}
```
