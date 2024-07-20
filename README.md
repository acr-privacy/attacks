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
