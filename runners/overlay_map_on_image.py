''' Overlays predicted slums and possibly true slums (if existent) onto the original satellite image, highlighting the aread used for calculations (i.e. the non-masked area).
Colour coding is: (i). Predicted Slums: RED, (ii). True Slums: BLUE, (iii). Predicted Slums = True Slums: MAGENTA, (iv). Area used for calculations: PALE GREEN
It requires the path to a configuration file that includes the following info:
        satellite_img_file:                 str, full path (and name) to the satellite image used for prediction (i.e. input_x)
        pred_slums_img_file:                str, full path (and name) to the file containing the predicted slum map (i.e. a single file reconstructed from the tiles)
        output_file:                        str, full path (and name) of the output file that will be saved
        mask_file:                          str, full path (and name) to the mask image, if used, (i.e. input_z) [default: None]
        true_slums_img_file:                str, full path (and name) to the file containing the true slums (i.e. the labels, input_y) [default: None]
        transparency:                       float (0,1): transparency of slum coloring [default: 0.5]
        prediction_mode:                    str ['binary', 'other'] indicates whether the model predicts yes/no slum locations 
                                            or distance to the nearest slum [default: 'binary']
ARGS:
    -c [--config_file]:             str, full path to the configuration file [default: \'reconstruct.yml\']
USAGE:
    >>> python3 overlay_map_on_image.py -c /path/to/overaly.yml
'''

import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import yaml
import json
import click
from pathlib import Path
try:
    from slumworldML.src.inspector import overlay
except Exception as Error:
    try:
        from src.inspector import overlay
    except Exception as Error2:
        from ..src.inspector import overlay

@click.command()
@click.option('-c', '--config_file', default=None, help='Path to the configuration file')
@click.option('--satellite_img_file', default=None, help='Raw (i.e. not tiled ) satellite image.')
@click.option('--pred_slums_img_file', default=None, help='Predicted slums reconstructed map file.')
@click.option('--mask_file', default=None, help='Mask image file.')
@click.option('--true_slums_img_file', default=None, help='Full path to the raw (i.e. not tilled) slum map image.')
@click.option('--output_file', default=None, help= 'Full filepath to the output map overlay.')
@click.option('--transparency', default=0.5, help='Transparency (%) of the overlay [0-1].')
@click.option('--binary', default=True, help='If using binary prediciton mode (True, not used).')
@click.help_option('-h', '--help')
def main(config_file, satellite_img_file, pred_slums_img_file, output_file, mask_file,
         true_slums_img_file, transparency, binary):
    if config_file:
        if os.path.exists(config_file):
            with open(config_file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            try:
                overlay(satellite_img_file=cfg['satellite_img_file'], 
                        pred_slums_img_file=cfg['pred_slums_img_file'], 
                        output_file=cfg['output_file'], 
                        mask_file=cfg['mask_file'], 
                        true_slums_img_file=cfg['true_slums_img_file'], 
                        transparency=cfg['transparency'], 
                        prediction_mode='binary')
                sys.exit(0)
            except Exception as Error:
                print("Error during overlay operation! Error log:", Error)
                sys.exit(2)
        else:
            print("Config file not found:", Error)
            sys.exit(1)
    try:
        overlay(satellite_img_file=satellite_img_file, 
                pred_slums_img_file=pred_slums_img_file, 
                output_file=output_file, 
                mask_file=mask_file, 
                true_slums_img_file=true_slums_img_file, 
                transparency=transparency, 
                prediction_mode='binary')
    except Exception as Error:
        print("Error during overlay operation! Error log:", Error)
        sys.exit(2)
    

if __name__ == "__main__":
    main()