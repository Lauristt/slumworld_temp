''' Tile an image and produce appropriate train-validation-test balancing dataset csv files.
It assumes that files are located in the base_dir = '/global/scratch/users/mksifaki/data/'.
It requires the name of an input directory with raw files and the name of the folder to store the results 
(will be created if non existent).
ARGS:
    --config [-c]:          str, path to a configuration file that provides values for all input parameters 
                            if supplied any other command line parameter will be ignored
    --input_dir [-i]:       str, the input directory name holding the raw files, input_x (and, potentially, input_y, input_z)
                            (located inside BASEPATH/raw/)
    --output_path [-i]:     str, the output directory name to put the tiled files (it will be created inside BASEPATH/tiled/)
    --tile_size [-t]:       int, the required tile size to tile into [default: 512]
    --split [-s]:           3*float, 3 floats separated by whitespavce for the [train, validation, test] set split fractions. [default: 0.7 0.15 0.15 ]
    --base_path [-b]:       str, the name of the basepath that holds the data [default:'/global/scratch/users/mksifaki/data/']
    --labels2binary:        boolean, if true [default] then the labels will be binarized, else the signed distances will be retained
USAGE:
    >>> INPUT_PATH = '/raw/MUL_MS_Round_105'
    >>> OUTPUT_PATH = '/tiled/MUL_MS_Round_105_TileSize_512'
    # tile image to a dataset with a train, validation with and test set split of  70%-15%-15% with tilesize 512
    >>> python3 tile_standard.py -i INPUT_PATH -o OUTPUT_PATH 
    # tile image to a dataset with a train, validation with and test set split of  80%-20%-0% with tilesize 256
    >>> python3 tile_standard.py -i INPUT_PATH -o OUTPUT_PATH -n 5 -s 0.8 0.2 0.0 -t 256
    # provide all parameters in a configuration file
    >>> python3 tile_standard.py -c ../config/tileMS.yml
'''

import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import math
import yaml
from pathlib import Path
import argparse
try: 
    from slumworldML.src.cnn_tiler import CNNTiler
    from slumworldML.src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
    from slumworldML.src.custom_transformations import BinarizeLabels
except Exception as Err1:
    try:
        from src.cnn_tiler import CNNTiler
        from src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
        from src.custom_transformations import BinarizeLabels
    except Exception as Err2:
        from ..src.cnn_tiler import CNNTiler
        from ..src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
        from ..src.custom_transformations import BinarizeLabels

def run(args):

    INPUTPATH = Path(args['input_dir'])
    assert INPUTPATH.is_dir(), f"Provided input path {INPUTPATH}  is not a directory. Exiting ..."
    OUTPUT_PATH= Path(args['output_dir'])
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    x_input_path = INPUTPATH/"input_x.png"
    y_input_path = INPUTPATH/"input_y.png"
    mask = INPUTPATH/"input_z.png"
    if not mask.exists():
        mask = None
    if not y_input_path.exists():
        y_input_path = None
    tiler = CNNTiler(tile_size=args['tile_size'], save_path=OUTPUT_PATH)
    print("Starting tiling operation...")

    tiler.create_standard_dataset(x_input_path=x_input_path,
                                  y_input_path=y_input_path, 
                                  training_split=args['split'],
                                  mask=mask,
                                  labels2binary=args['labels2binary'])
    
    if (y_input_path is not None) and (args['split'][0] > 0):
        print("Finished tiling and dataset creation.\nCalculating balancing statistics...")
        transformation = TRAINING_TRANSFORMS_BASIC
        if not args['labels2binary']:
            transformation['joint_transforms'].insert(0, BinarizeLabels())
        transformation = create_transform(TRAINING_TRANSFORMS_BASIC, mean=[0,0,0], std=[1,1,1]) 
        tiler.calculate_tile_statistics(transformation=transformation,
                                        num_of_samples=100,
                                        dataset_name='dataset.csv',
                                        save_csv=True)
    print("Successfully completed all operations.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('-c', '--from_config', default=None, type=str, help="Load all parameters from a config file.") 
    parser.add_argument('-i', '--input_dir', type=str, help="Input path.") 
    parser.add_argument('-o', '--output_dir', type=str, help="The output directory name to put the tiled files (it will be created inside BASEPATH/tiled/).", required=False)
    parser.add_argument('-s', '--split', nargs=3, type=float, default=[0.7,0.15,0.15], help="Train, validation, test set split fractions.") 
    parser.add_argument('-t', '--tile_size', type=int, default=512, help="The required tile size to tile into.")
    parser.add_argument('-b', '--base_path', type=str, required=False, default='/global/scratch/users/mksifaki/data/', help="Basepath that holds the data (raw and tiled).")
    parser.add_argument('--labels2binary', type=bool, default=True, help="Whether the labels should be binarized (default) or signed distances should be kept.")
    args = vars(parser.parse_args())
    if args['from_config']:
        import yaml
        try:
            with open(args['from_config'], 'r') as fin:
                config = yaml.safe_load(fin)
            print(f"Loaded parameters from cofiguration file {args['from_config']}")
            for key, value in config.items():
                args[key] = value
        except Exception as Error:
            print(f"Error trying to load configuration file {args['from_config']}.", Error)
    for argument in ['input_dir', 'output_dir']:
        if (args[argument] is None) and (args['from_config'] is None):
            print(f"Error! {argument} not provided or is None. Aborting ...")
            sys.exit(1)
    assert math.isclose(sum(args['split']), 1.0), f"Split fractions should sum to 1.0 but supplied values {args['split']} sum to {sum(args['split']) }! Aborting operation..."

    run(args)