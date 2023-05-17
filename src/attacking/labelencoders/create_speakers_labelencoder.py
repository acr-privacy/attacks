#!/usr/bin/env python
import click
from sklearn import preprocessing
from joblib import dump, load

@click.command()
@click.option('--min_label', default=1, help='min label id')
@click.option('--max_label', default=10, help='max label id')
@click.argument('filename')
def main(min_label, max_label, filename):
    le = create_labelencoder(min_label, max_label)
    save_labelencoder(le, filename)

def create_labelencoder(min_label, max_label):
    le = preprocessing.LabelEncoder()
    labels = [f'{i:02d}' for i in range(min_label, max_label+1)]
    le.fit(labels)
    return le

def save_labelencoder(labelencoder, dst_file):
    dump(labelencoder, dst_file)

def test_labelencoder(labelencoder):
    labels = ['08', '08', '02', '03']
    print(labelencoder.transform(labels))

if __name__ == '__main__':
    main()
