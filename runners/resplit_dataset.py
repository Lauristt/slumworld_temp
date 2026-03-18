''' Loads an existing dataset.csv file and produces a different train-validation-test split, recalculating normalization statistics in the process.
    Creates and saves new dataset.csv and dataset.json files.
ARGS:
    --config [-c]:      str, full path to a configuration file [default: ../config/resplit_dataset.yml] containing the following fields:
                        dataset_csv:            str, existing dataset.csv file that will be resplit
                        split_ratio:            str, train-val-test split [default:[0.6,0.2,0.2]]
                        save_location:          str, full path (and filename) to the location that the combined dataset and json will be saved
                                                if left empty the script will overwrite the existing files [default:None]
                              
USAGE:
    >>> cd runners
    >>> python3 resplit_dataset.py -c ../config/respit_dataset.yml 
'''
import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
from pathlib import Path
import yaml
import argparse
import pandas as pd
try:
    from slumworldML.src.cnn_tiler import resplit_dataset
except Exception as Error:
    try:
        from src.cnn_tiler import resplit_dataset
    except Exception as Error2:
        from ..src.cnn_tiler import resplit_dataset


def run(args):
    
    try:
        resplit_dataset(args['dataset_csv'], 
                        args['split_ratio'], 
                        args['save_location'])
    except Exception as Error:
        print("Error encountered, aborting... ", Error)
        sys.exit(2)
    print("Process completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('-c', '--config_file', type=str, default=Path.home()/'slumworldML'/'config'/'resplit_dataset.yml', 
                        help="Configuration yml file for running the script. ")

    args = vars(parser.parse_args())

    try:
        with open(args['config_file'], 'r') as fin:
            config = yaml.safe_load(fin)
            print("Loaded Configuration:\n",config)
    except Exception as Error:
        print("Error! Could not load configuration file. Reason:", Error)
        sys.exit(1)

    run(config)