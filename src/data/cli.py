from pathlib import Path

import click

from data.util import Extractor, FileDownloader, file_md5_equals_expected_md5


@click.command("prepare")
@click.option("-o", "--outdir", type=Path, help="Path to dataset")
def prepare_cmd(outdir: Path):
    BASE_URL = "https://www.ias.cs.tu-bs.de/publications"
    ARCHIVE_NAME = "acr_privacy_attacks_datasets.tar.gz"
    ARCHIVE_MD5 = "ec7e0b001db2e0336846d1db834f259d"

    create_dir_or_die(outdir)

    file_url = "/".join((BASE_URL, ARCHIVE_NAME))
    archive_path = outdir.joinpath(ARCHIVE_NAME)

    with FileDownloader(file_url) as dl:
        dl.download_to_file_with_progressbar(archive_path)

    click.echo("Verifying checksum.. ", nl=False)
    if file_md5_equals_expected_md5(archive_path, ARCHIVE_MD5):
        click.echo("OK")
    else:
        click.echo("FAILED")
        exit(1)

    with Extractor(archive_path) as ex:
        ex.extract_to_dir_with_progressbar(outdir)

    click.echo("Deleting archive.. ", nl=False)
    archive_path.unlink()
    click.echo("DONE")


@click.group()
def cli():
    pass


cli.add_command(prepare_cmd)


def create_dir_or_die(outdir: Path):
    try:
        outdir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        click.echo(
            (
                f"Could not create target directory '{outdir}'. "
                "Directory already exists."
            ),
            err=True,
        )
        exit(1)
