import os
from pathlib import Path

import click
import pandas as pd

from modeling.train_speaker_model import run_speaker_experiment
from modeling.train_speech_model import run_speech_experiment
from modeling.train_words_model import run_words_experiment
from modeling.transformer import create_model


@click.command("words-experiment")
@click.option("--datadir", type=Path, help="dataset path")
@click.option("--outdir", type=Path, help="result path")
@click.option(
    "--visible-devices", type=str, help="override CUDA_VISIBLE_DEVICES"
)
@click.option(
    "--embedding-dim",
    type=int,
    default=128,
    help="number of embedding dimensions",
)
@click.option(
    "--embedding-bias", is_flag=True, help="number of embedding dimensions"
)
@click.option(
    "--embedding-dropout",
    type=float,
    default=0.1,
    help="dropout ratio of embedding layer",
)
@click.option(
    "--n-encoders", type=int, default=4, help="number of encoder blocks"
)
@click.option(
    "--n-heads", type=int, default=4, help="number of heads per encoder block"
)
@click.option(
    "--mlp-factor",
    type=int,
    default=1,
    help=(
        "hidden layer of mlp encoder layer will have "
        " mlp_factor * emb_dim neurons"
    ),
)
@click.option(
    "--encoder-dropout",
    type=float,
    default=0.1,
    help="dropout ratio of encoder blocks",
)
@click.option(
    "--sd-proba",
    type=float,
    default=0.1,
    help="maximum layer skip probability",
)
@click.option(
    "--post-encoder-norm",
    is_flag=True,
    help="enable layer norm after encoder blocks",
)
@click.option(
    "--method",
    type=click.Choice(["acrcloud", "sonnleitner", "zapr_alg1", "zapr_alg2"]),
    help="select fingerprint method",
)
@click.option("--batch-size", type=int, default=32, help="training batch size")
@click.option(
    "--n-epochs", type=int, default=300, help="number of total training epochs"
)
@click.option(
    "--n-decay-epochs",
    type=int,
    default=300,
    help="number of epochs per cosine decay cycle",
)
@click.option(
    "--lr-init", type=float, default=1e-3, help="initial learning rate"
)
@click.option(
    "--lr-min", type=float, default=0.0, help="minimum learning rate"
)
@click.option(
    "--weight-decay",
    type=float,
    default=1e-4,
)
@click.option(
    "--label-smoothing", type=float, default=1e-1, help="minimum learning rate"
)
@click.option(
    "--seq-len",
    type=int,
    help="maximum fingerprint/sequence length handled by the model",
)
@click.option(
    "--n-bits", type=int, help="ZAPR2 ONLY: number of bits per subfingerprint"
)
def words_experiment_cmd(
    datadir: Path,
    outdir: Path,
    visible_devices: str,
    embedding_dim: int,
    embedding_bias: bool,
    embedding_dropout: float,
    n_encoders: int,
    n_heads: int,
    mlp_factor: int,
    encoder_dropout: float,
    sd_proba: float,
    post_encoder_norm: bool,
    method: str,
    batch_size: int,
    n_epochs: int,
    n_decay_epochs: int,
    lr_init: float,
    lr_min: float,
    weight_decay: float,
    label_smoothing: float,
    seq_len: int,
    n_bits: int,
):
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    outdir.mkdir(parents=True, exist_ok=True)

    model_params = {
        "n_classes": 10,
        "embedding_dim": embedding_dim,
        "embedding_bias": embedding_bias,
        "embedding_dropout": embedding_dropout,
        "n_encoders": n_encoders,
        "n_heads": n_heads,
        "mlp_factor": mlp_factor,
        "encoder_dropout": encoder_dropout,
        "sd_proba": sd_proba,
        "post_encoder_norm": post_encoder_norm,
        "seq_len": seq_len,
    }

    if method == "acrcloud":
        model_params["n_bits"] = 64
    elif method == "sonnleitner":
        model_params["n_bits"] = 19
    elif method == "zapr_alg1":
        model_params["n_bits"] = 32
    elif method == "zapr_alg2":
        model_params["n_bits"] = n_bits
    else:
        raise NotImplementedError

    model = create_model(**model_params)

    experiment_params = {
        "model": model,
        "datadir": datadir,
        "outdir": outdir,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_decay_epochs": n_decay_epochs,
        "lr_init": lr_init,
        "lr_min": lr_min,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "method": method,
    }

    params = {**model_params, **experiment_params}
    params.pop("model")
    params.pop("outdir")
    pd.DataFrame(params, index=[0]).to_csv(outdir / "params.csv", index=False)

    run_words_experiment(**experiment_params)


@click.command("speaker-experiment")
@click.option("--datadir", type=Path, help="dataset path")
@click.option("--outdir", type=Path, help="result path")
@click.option(
    "--visible-devices", type=str, help="override CUDA_VISIBLE_DEVICES"
)
@click.option(
    "--embedding-dim",
    type=int,
    default=128,
    help="number of embedding dimensions",
)
@click.option(
    "--embedding-bias", is_flag=True, help="number of embedding dimensions"
)
@click.option(
    "--embedding-dropout",
    type=float,
    default=0.1,
    help="dropout ratio of embedding layer",
)
@click.option(
    "--n-encoders", type=int, default=4, help="number of encoder blocks"
)
@click.option(
    "--n-heads", type=int, default=4, help="number of heads per encoder block"
)
@click.option(
    "--mlp-factor",
    type=int,
    default=1,
    help=(
        "hidden layer of mlp encoder layer will have "
        " mlp_factor * emb_dim neurons"
    ),
)
@click.option(
    "--encoder-dropout",
    type=float,
    default=0.1,
    help="dropout ratio of encoder blocks",
)
@click.option(
    "--sd-proba",
    type=float,
    default=0.1,
    help="maximum layer skip probability",
)
@click.option(
    "--post-encoder-norm",
    is_flag=True,
    help="enable layer norm after encoder blocks",
)
@click.option(
    "--method",
    type=click.Choice(["acrcloud", "sonnleitner", "zapr_alg1", "zapr_alg2"]),
    help="select fingerprint method",
)
@click.option("--batch-size", type=int, default=32, help="training batch size")
@click.option(
    "--n-epochs", type=int, default=300, help="number of total training epochs"
)
@click.option(
    "--n-decay-epochs",
    type=int,
    default=300,
    help="number of epochs per cosine decay cycle",
)
@click.option(
    "--lr-init", type=float, default=1e-3, help="initial learning rate"
)
@click.option(
    "--lr-min", type=float, default=0.0, help="minimum learning rate"
)
@click.option(
    "--weight-decay",
    type=float,
    default=1e-4,
)
@click.option(
    "--label-smoothing", type=float, default=1e-1, help="minimum learning rate"
)
@click.option(
    "--split",
    type=str,
    default="testing",
    help="Chose the sub dataset (training or testing)",
)
@click.option(
    "--seq-len",
    type=int,
    help="maximum fingerprint/sequence length handled by the model",
)
@click.option(
    "--n-bits", type=int, help="ZAPR2 ONLY: number of bits per subfingerprint"
)
def speaker_experiment_cmd(
    datadir: Path,
    outdir: Path,
    visible_devices: str,
    embedding_dim: int,
    embedding_bias: bool,
    embedding_dropout: float,
    n_encoders: int,
    n_heads: int,
    mlp_factor: int,
    encoder_dropout: float,
    sd_proba: float,
    post_encoder_norm: bool,
    method: str,
    batch_size: int,
    n_epochs: int,
    n_decay_epochs: int,
    lr_init: float,
    lr_min: float,
    weight_decay: float,
    label_smoothing: float,
    split: str,
    seq_len: int,
    n_bits: int,
):
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    outdir.mkdir(parents=True, exist_ok=True)

    model_params = {
        "n_classes": 40,
        "embedding_dim": embedding_dim,
        "embedding_bias": embedding_bias,
        "embedding_dropout": embedding_dropout,
        "n_encoders": n_encoders,
        "n_heads": n_heads,
        "mlp_factor": mlp_factor,
        "encoder_dropout": encoder_dropout,
        "sd_proba": sd_proba,
        "post_encoder_norm": post_encoder_norm,
        "seq_len": seq_len,
    }

    if method == "acrcloud":
        model_params["n_bits"] = 64
    elif method == "sonnleitner":
        model_params["n_bits"] = 19
    elif method == "zapr_alg1":
        model_params["n_bits"] = 32
    elif method == "zapr_alg2":
        model_params["n_bits"] = n_bits
    else:
        raise NotImplementedError

    model = create_model(**model_params)

    experiment_params = {
        "model": model,
        "datadir": datadir,
        "outdir": outdir,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_decay_epochs": n_decay_epochs,
        "lr_init": lr_init,
        "lr_min": lr_min,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "method": method,
        "split": split,
    }

    params = {**model_params, **experiment_params}
    params.pop("model")
    params.pop("outdir")
    pd.DataFrame(params, index=[0]).to_csv(outdir / "params.csv", index=False)

    run_speaker_experiment(**experiment_params)


@click.command("speech-experiment")
@click.option("--datadir", type=Path, help="dataset path")
@click.option("--outdir", type=Path, help="result path")
@click.option(
    "--visible-devices", type=str, help="override CUDA_VISIBLE_DEVICES"
)
@click.option(
    "--embedding-dim",
    type=int,
    default=128,
    help="number of embedding dimensions",
)
@click.option(
    "--embedding-bias", is_flag=True, help="number of embedding dimensions"
)
@click.option(
    "--embedding-dropout",
    type=float,
    default=0.1,
    help="dropout ratio of embedding layer",
)
@click.option(
    "--n-encoders", type=int, default=4, help="number of encoder blocks"
)
@click.option(
    "--n-heads", type=int, default=4, help="number of heads per encoder block"
)
@click.option(
    "--mlp-factor",
    type=int,
    default=1,
    help=(
        "hidden layer of mlp encoder layer will have "
        " mlp_factor * emb_dim neurons"
    ),
)
@click.option(
    "--encoder-dropout",
    type=float,
    default=0.1,
    help="dropout ratio of encoder blocks",
)
@click.option(
    "--sd-proba",
    type=float,
    default=0.1,
    help="maximum layer skip probability",
)
@click.option(
    "--post-encoder-norm",
    is_flag=True,
    help="enable layer norm after encoder blocks",
)
@click.option(
    "--method",
    type=click.Choice(["acrcloud", "sonnleitner", "zapr_alg1", "zapr_alg2"]),
    help="select fingerprint method",
)
@click.option("--batch-size", type=int, default=32, help="training batch size")
@click.option(
    "--n-epochs", type=int, default=300, help="number of total training epochs"
)
@click.option(
    "--n-decay-epochs",
    type=int,
    default=300,
    help="number of epochs per cosine decay cycle",
)
@click.option(
    "--lr-init", type=float, default=1e-3, help="initial learning rate"
)
@click.option(
    "--lr-min", type=float, default=0.0, help="minimum learning rate"
)
@click.option(
    "--weight-decay",
    type=float,
    default=1e-4,
)
@click.option(
    "--label-smoothing", type=float, default=1e-1, help="minimum learning rate"
)
@click.option(
    "--seq-len",
    type=int,
    help="maximum fingerprint/sequence length handled by the model",
)
@click.option(
    "--n-bits", type=int, help="ZAPR2 ONLY: number of bits per subfingerprint"
)
def speech_experiment_cmd(
    datadir: Path,
    outdir: Path,
    visible_devices: str,
    embedding_dim: int,
    embedding_bias: bool,
    embedding_dropout: float,
    n_encoders: int,
    n_heads: int,
    mlp_factor: int,
    encoder_dropout: float,
    sd_proba: float,
    post_encoder_norm: bool,
    method: str,
    batch_size: int,
    n_epochs: int,
    n_decay_epochs: int,
    lr_init: float,
    lr_min: float,
    weight_decay: float,
    label_smoothing: float,
    seq_len: int,
    n_bits: int,
):
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    outdir.mkdir(parents=True, exist_ok=True)

    model_params = {
        "n_classes": 2,
        "embedding_dim": embedding_dim,
        "embedding_bias": embedding_bias,
        "embedding_dropout": embedding_dropout,
        "n_encoders": n_encoders,
        "n_heads": n_heads,
        "mlp_factor": mlp_factor,
        "encoder_dropout": encoder_dropout,
        "sd_proba": sd_proba,
        "post_encoder_norm": post_encoder_norm,
        "seq_len": seq_len,
    }

    if method == "acrcloud":
        model_params["n_bits"] = 64
    elif method == "sonnleitner":
        model_params["n_bits"] = 19
    elif method == "zapr_alg1":
        model_params["n_bits"] = 32
    elif method == "zapr_alg2":
        model_params["n_bits"] = n_bits
    else:
        raise NotImplementedError

    model = create_model(**model_params)

    experiment_params = {
        "model": model,
        "datadir": datadir,
        "outdir": outdir,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_decay_epochs": n_decay_epochs,
        "lr_init": lr_init,
        "lr_min": lr_min,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "method": method,
    }

    params = {**model_params, **experiment_params}
    params.pop("model")
    params.pop("outdir")
    pd.DataFrame(params, index=[0]).to_csv(outdir / "params.csv", index=False)

    run_speech_experiment(**experiment_params)


@click.group()
def cli():
    pass


cli.add_command(words_experiment_cmd)
cli.add_command(speaker_experiment_cmd)
cli.add_command(speech_experiment_cmd)
