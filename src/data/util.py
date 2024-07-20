import hashlib
import re
import tarfile
from collections import namedtuple
from pathlib import Path

import click
import pandas as pd
import requests
from tqdm import tqdm


class FileDownloader:
    def __init__(self, url: str):
        self.url = url

    def __enter__(self):
        self.response = requests.get(self.url, stream=True)
        return self

    def __exit__(self, *args):
        pass

    def download_to_file_with_progressbar(self, file_path: Path):
        description = "Downloading file.."
        click.echo(description, nl=False)

        content_length = self.response_content_length(self.response)

        with tqdm(
            desc=description,
            unit="B",
            unit_scale=True,
            leave=False,
            total=content_length,
            bar_format="{desc} {percentage:3.0f}%|{bar}{r_bar}",
        ) as bar:
            CHUNK_SIZE = 1024 * 1024
            chunks = self.response.iter_content(chunk_size=CHUNK_SIZE)

            with file_path.open("wb") as file:
                for chunk in chunks:
                    file.write(chunk)
                    bar.update(len(chunk))

        click.echo(f"{description} DONE")

    @staticmethod
    def response_content_length(response: requests.Response) -> int:
        return int(response.headers.get("content-length"))


class Extractor:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.tar = tarfile.open(self.path)
        return self

    def __exit__(self, *args):
        self.tar.close()

    def extract_to_dir_with_progressbar(self, dst_dir: Path):
        description = "Extracting archive.."
        click.echo(description, nl=False)

        with tqdm(
            desc=description,
            unit="B",
            unit_scale=True,
            leave=False,
            total=self.decompressed_size(),
            bar_format="{desc} {percentage:3.0f}%|{bar}{r_bar}",
        ) as bar:
            for member in self.tar:
                self.tar.extract(member, dst_dir)
                bar.update(Extractor.decompressed_member_size(member))

        click.echo(f"{description} DONE")

    def decompressed_size(self) -> int:
        return sum(map(self.decompressed_member_size, self.tar))

    @staticmethod
    def decompressed_member_size(member: tarfile.TarInfo) -> int:
        return member.get_info()["size"]


class Dataset:
    @classmethod
    def load_metadata(cls, p: Path) -> pd.DataFrame:
        sample_paths = p.rglob("*.fprint")
        idx, meta = zip(*map(cls.path_to_meta, sample_paths))

        df = pd.DataFrame(meta, index=idx)
        df.sort_index(inplace=True)

        return df

    @classmethod
    def path_to_meta(cls, p: Path) -> (str, namedtuple):
        raise NotImplementedError


class SpeechVsMusicDataset(Dataset):
    sample_counter = 0

    SampleMeta = namedtuple("SampleMeta", ["split", "label", "path"])

    @classmethod
    def path_to_meta(cls, p: Path) -> (str, SampleMeta):
        pattern = re.compile(
            (
                r".*/(?P<split>training|validation|testing)"
                r"/(fma_small|.*3secs)/.*\.fprint"
            )
        )

        match = pattern.match(str(p))

        idx = cls.sample_counter
        cls.sample_counter += 1

        meta = cls.SampleMeta(
            **match.groupdict(),
            label=cls.parse_label(p),
            path=str(p),
        )

        return idx, meta

    @classmethod
    def parse_label(cls, p: Path) -> str:
        if p.match("fma_small/*/*/*.fprint"):
            return "music"
        elif p.match("*3secs/*/*.fprint"):
            return "speech"


class SpeakersDataset(Dataset):
    SampleMeta = namedtuple(
        "SampleMeta",
        ["speaker_id", "chapter_id", "utterance", "chunk", "split", "path"],
    )

    @classmethod
    def path_to_meta(cls, p: Path) -> (str, SampleMeta):
        pattern = re.compile(
            (
                r"(?P<speaker_id>\d+)"
                r"-(?P<chapter_id>\d+)"
                r"-(?P<utterance>\d+)"
                r"-chunk-(?P<chunk>\d+)$"
            )
        )

        match = pattern.match(p.stem)

        idx = "-".join(match.groups())

        meta = cls.SampleMeta(
            **match.groupdict(),
            split=p.parent.name,
            path=str(p),
        )

        return idx, meta


class WordsDataset(Dataset):
    SampleMeta = namedtuple("SampleMeta", ["unknown", "word", "split", "path"])

    @classmethod
    def path_to_meta(cls, p: Path) -> (str, SampleMeta):
        pattern = re.compile((r"(?P<unknown>\d+)" r"_(?P<word>.+)$"))

        match = pattern.match(p.stem)

        idx = "-".join(match.groups())

        meta = cls.SampleMeta(
            **match.groupdict(),
            split=p.parent.name,
            path=str(p),
        )

        return idx, meta


def file_md5_equals_expected_md5(file: Path, expected_md5: str) -> bool:
    calculated_md5 = hashlib.md5(file.read_bytes()).hexdigest()
    return calculated_md5 == expected_md5
