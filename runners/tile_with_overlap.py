''' Tile an image for inference using 50% tile overlap and extended (1/4 tile) padding on top and left.
ARGS:
    --from_config [-c]:         str, path to the config file to use for loading parameters (will ignore all other command line parameters)
    --input_image_path [-i]:    str, the input (raw) satellite image that will be tiled
    --output_folder [-o]:       str, the output directory name to put the tiled files (it will be created if non-existent)
                                The output folder SHOULD NOT CONTAIN ANY IMAGES in it else the operation will be aborted (to ensure no accidental data loss).
    --tile_size [-t]:           int, the required tile size to tile into [default: 512]
    --info_json_folder [-j]:      Directory to save the an info.json file that is required for reconstruction of the original image. 
                                Default ['output_folder']
    --is_label [-l]:            Needs to be added (without any argument) when tiling labels images (i.e. single channel)
USAGE:
    # from an appropriate config file
    >>> python3 tile_with_overlap.py -c ../config/tile_with_overlap.yml
    # from command line
    >>> INPUT_PATH = '/global/scratch/users/mksifaki/data/raw/MUL_MS_Round_105/input_x.png'
    >>> OUTPUT_PATH = '/global/scratch/users/mksifaki/data/tiled/MUL_MS_Round_105_with_Overlap'
    # tile image to a dataset with a train, validation with and test set split of  70%-15%-15% with tilesize 512
    >>> python3 tile_with_overlap.py -i INPUT_PATH -o OUTPUT_PATH -t 512 -j
'''

import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import yaml
from pathlib import Path
import argparse
try:
    from slumworldML.src.overlap_tiler import OverlapTiler
    from slumworldML.src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
except Exception as Err1:
    try:
        from src.overlap_tiler import OverlapTiler
        from src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
    except Exception as Err2:
        from ..src.overlap_tiler import OverlapTiler
        from ..src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC

def run(args):
    os.makedirs(args['output_folder'], exist_ok=True)
    tiler = OverlapTiler(input_image_path=args['input_image_path'], 
                         output_folder=args['output_folder'], 
                         tile_size=args['tile_size'], 
                         info_json_folder=args['info_json_folder'], 
                         is_label_image=args['is_label'])
    print("Starting tiling operation...")

    tiler.tile_and_save()

    print("Successfully completed tiling operation.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)

    parser.add_argument('-c', '--from_config', default=None, type=str, help="Load all parameters from a config file.") 
    parser.add_argument('-i', '--input_image_path', type=str, help="Path to the satellite image that will be tiled.") 
    parser.add_argument('-o', '--output_folder', type=str, help="The output directory name to put the tiled files (it will be created if non existent).")
    parser.add_argument('-t', '--tile_size', type=int, default=512, help="The required tile size to tile into [default:512].")
    parser.add_argument('-j', '--info_json_folder', type=str, required=False, default=None, 
                        help="Directory to save the an info.json file that is required for reconstruction of the original image. Default ['output_folder']")
    parser.add_argument('-l', '--is_label', action='store_true', help="If provided the flag signifies that we are tiling a label image (single channel).", required=False)

    args = vars(parser.parse_args())
    if args['from_config']:
        import yaml
        try:
            with open(args['from_config'], 'r') as fin:
                config = yaml.safe_load(fin)
            print(f"Loaded parameters from cofiguration file {args['from_config']}")
            print(config)
            for key, value in config.items():
                args[key] = value
        except Exception as Error:
            print(f"Error trying to load configuration file {args['from_config']}.", Error)
    for argument in ['input_image_path', 'output_folder']:
        if args[argument] is None:
            print(f"Error! {argument} not provided or is None. Aborting ...")
            sys.exit(1)
    run(args)

