import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import yaml
import json
import argparse
from skimage import io
from pathlib import Path
try:
    from slumworldML.src.cnn_tiler import CNNTiler
except Exception as Error:
    try:
        from ..src.cnn_tiler import CNNTiler
    except Exception as Error2:
        try:
            from .src.cnn_tiler import CNNTiler
        except Exception as Error3:
            try:
                from src.cnn_tiler import CNNTiler
            except Exception as Error4:
                from cnn_tiler import CNNTiler

def main(config):
    def get_tile_size(path):
        '''load the first image to get the tile size'''
        img = io.imread(os.path.join(path,os.listdir(path)[0]))
        return img.shape[0]
    img_folder = str(Path(config['img_folder']).absolute())
    dataset_name = config['dataset_name']
    output_path = str(Path(config['output_path']).absolute())
    label_folder = str(Path(config['label_folder']).absolute())
    calculate_stats = config['calculate_stats']
    training_split = config['train_val_test_split']
    tile_size = get_tile_size(img_folder)
    try:
        tiler = CNNTiler(tile_size=tile_size)
        tiler.dataset_from_folder(img_folder=img_folder, 
                                  label_folder=label_folder, 
                                  training_split=training_split, 
                                  save_path=output_path, 
                                  dataset_name=dataset_name, 
                                  calculate_stats=calculate_stats)

    except Exception as Error:
        print("Error during map reconstrunction! Error type:", Error)
        sys.exit(2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-c', '--config_file', type=str, default=Path.home()/'slumworldML'/'config'/'dataset_from_folder.yml', help="For generating dataset.csv training files from arbitrary sources. Default: \'~\/slumworldML\/config\/dataset_from_folder.yml\'")
    args = vars(parser.parse_args())
    try:
        with open(args['config_file'], 'r') as fin:
            config = yaml.safe_load(fin)
    except Exception as Error:
        print("Error! Could not load configuration file. Reason:", Error)
        sys.exit(1)

    main(config)