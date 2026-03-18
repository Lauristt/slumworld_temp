import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import yaml
import json
import argparse
from pathlib import Path
try:
    from slumworldML.src.cnn_tiler import CNNTiler
except Exception as Error:
    try:
        from src.cnn_tiler import CNNTiler
    except Exception as Error2:
        try:
            from .src.cnn_tiler import CNNTiler
        except Exception as Error3:
            from ..src.cnn_tiler import CNNTiler


def main(config):
    img_folder = str(Path(config['img_folder']).absolute())
    dataset_name = config['dataset_name']
    output_filepath = str(Path(config['output_path']).absolute())
    label_folder = str(Path(config['label_folder']).absolute())
    num_of_folds = config['num_of_folds']
    test_set_frac = config['test_set_frac']
    calculate_stats = config['calculate_stats']
    try:
        tiler = CNNTiler(tile_size=512)
        tiler.kfold_dataset_from_folder(img_folder=img_folder, label_folder=label_folder, 
                                        num_of_folds=num_of_folds, test_set_frac=test_set_frac, 
                                        save_path=output_filepath, dataset_name=dataset_name, 
                                        calculate_stats=calculate_stats)

    except Exception as Error:
        print("Error during map reconstrunction! Error type:", Error)
        sys.exit(2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-c', '--config_file', type=str, default=Path.home()/'slumworldML'/'config'/'kfold_dataset_from_folder.yml', help="For generating a k-fold-dataset.csv training files from arbitrary sources. Default: \'~\/slumworldML\/config\/kfold_dataset_from_folder.yml\'")
    args = vars(parser.parse_args())
    try:
        with open(args['config_file'], 'r') as fin:
            config = yaml.safe_load(fin)
    except Exception as Error:
        print("Error! Could not load configuration file. Reason:", Error)
        sys.exit(1)

    main(config)