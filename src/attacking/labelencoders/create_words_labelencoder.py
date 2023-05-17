#!/usr/bin/env python
import click
from sklearn import preprocessing
from joblib import dump

@click.command()
@click.argument('filename')
def main(filename):
    le = create_labelencoder()
    save_labelencoder(le, filename)
    test_labelencoder(le)


def create_labelencoder():
    word_list = ['backward', 'bed', 'bird', 'cat', 'dog',
                 'down', 'eight', 'five', 'follow', 'forward',
                 'four', 'go', 'happy', 'house', 'learn',
                 'left', 'marvin', 'nine', 'no', 'off',
                 'on', 'one', 'right', 'seven', 'sheila',
                 'six', 'stop', 'three', 'tree', 'two',
                 'up', 'visual', 'wow', 'yes', 'zero']
    le = preprocessing.LabelEncoder()
    le.fit(word_list)
    return le


def save_labelencoder(labelencoder, dst_file):
    dump(labelencoder, dst_file)


def test_labelencoder(labelencoder):
    labels = ['one', 'right', 'three', 'visual']
    print(labelencoder.transform(labels))


if __name__ == '__main__':
    main()
