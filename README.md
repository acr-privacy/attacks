# Quickstart

1. Install dependencies:

- python 3.10
- gnumake

Nix users can just run `nix develope`.


2. Install python dependencies:

If you have GPU support run

```bash
pip install .[gpu]
```

If you do not have GPU support run

```bash
pip install .[cpu]
```

If you want to reproduce our exact setup with GPU support run

```bash
pip install -r requirements.txt
```
In this setup we utilized the following hardware: 

- 1x RTX 2080 Ti
- Driver Version: 555.42.06
- CUDA Version: 12.5

3. Download and prepare dataset:

```bash
make dataset
```

4. Set up your environment:

```bash
export KERAS_BACKEND="tensorflow";
CUDA_VISIBLE_DEVICES="XXX"
```

5. Run experiment:

```bash
make experiments
```
