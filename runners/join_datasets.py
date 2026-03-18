''' Join together two or more datasets to produce a combosite dataset.csv file that can be used for training/validation/testing/inference
Requires a new line delimited list of the full path to the individual dataset.csv files that will be combined and the location
(and name) that the joint dataset will be saved to.

ARGS:
    --dataset_list [-d]:      file, full path to a new line delimited file holding the location of the individual \'dataset.csv\' 
                              files that will be combined [default: ../config/datasets.lst]
    --save_path [-s]:         str, full path  to the location ((and name of the file)) that the joint \'combined_dataset.csv\' 
                              file should be stored.
USAGE:
    >>> cd runners
    >>> python3 join_datasets.py -d ../config/datasets.lst -s /global/scratch/users/mksifaki/data/tiled/combined/MD_MS_combined.csv
'''
import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
from pathlib import Path
import json
import argparse
import pandas as pd
try:
    from slumworldML.src.cnn_tiler import CNNTiler, combine_datasets
except Exception as Error:
    try:
        from src.cnn_tiler import CNNTiler, combine_datasets
    except Exception as Error2:
        from ..src.cnn_tiler import CNNTiler, combine_datasets


def run(args):

    try:
        with open(args['dataset_list'], 'r') as fin:
            paths = fin.readlines()
        paths = [p.strip() for p in paths]
        paths = [p for p in paths if p.endswith('.csv')]
        assert len(paths) > 1, f"You must supply at least 2 dataset.csv files for merging. Found only {len(paths)} csv files."
        os.makedirs(os.path.dirname(args['save_path']),exist_ok=True)
    except Exception as Error:
        print("Error! Could not load dataset list:", Error)
        sys.exit(1)

    combine_datasets(paths, args['save_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('-d', '--dataset_list', type=str, default=Path('.')/'config'/'datasets.lst', help="Full path to a new line delimited file holding the location of the individual \'dataset.csv\' files that will be joint.") 
    parser.add_argument('-s', '--save_path', type=str, required=True, default=None, help="Full path  to the location (and name of the file) that the joint \`combined_dataset.csv\` file should be stored.")

    args = vars(parser.parse_args())

    run(args)