'''Generates a .gif animation produced from all images in a folder that match a specified prefix.
   It is typically used to produce animations from the prediction_samples folder and hence inspect the 
   training process.
   The user has to supply the location of a configuration file which should point to the location of the 
   image folder, the prefix of the images of interest and the desired animaton frequency (fps).
   CONFIG file (yml) fields:
        input_images_path:                  str, path to the raw satellite image we are predicting on (input_x)
        output_folder:                      str, filename and path to the folder where the animation will be saved
        image_name:                         str, the name of the images that will be used to produce the animation
                                            e.g. to get all images 'epoch_00XXX_12800_6144.png' supply '12800_6144.png'
        fps:                                int, animation speed in frames per second
    
    USAGE:
        >>> python3 animate_training_samples.py -c ../config/animate.yml

'''

import sys
import os
import yaml
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
from pathlib import Path
import argparse
try:
    from slumworldML.src.inspector import animate
except Exception as Error:
    try:
        from src.inspector import animate
    except Exception as Error2:
        from ..src.inspector import animate


def run(args):
    if not os.path.exists(args['output_folder']):
        os.makedirs(args['output_folder'], exist_ok=True)
    try:
        p = Path(args['input_images_path'])
        images = [f.name for f in p.iterdir() if f.name[-4:] in ['.jpg', '.png']]
        image_list = sorted([ f for f in images if args['image_name'] in f], reverse=False)
        print("Started processing...")
        animate(input_path=args['input_images_path'],
                image_list=image_list, 
                output_filename=os.path.join(args['output_folder'],'animation_of_tile_'+args['image_name']), 
                fps=args['fps'],
                )
    except Exception as Error:
        print("Error during processing! Error log:", Error)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]), formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-c', '--config_file', type=str, default=Path.home()/'slumworldML'/'config'/'animate.yml', help="Configuration yml file for running the animate_training_samples script."\
        "Default: \'~\/slumworldML\/config\/animate.yml\'")
    args = vars(parser.parse_args())
    try:
        with open(args['config_file'], 'r') as fin:
            config = yaml.safe_load(fin)
    except Exception as Error:
        print("Error! Could not load configuration file. Reason:", Error)
        sys.exit(1)

    run(config)