''' Script for reconstructing a slum map from prediction tiles.
It requires the path to a configuration file that includes the following info:
    image_dir:                      str, the full path to the directory holding the prediction tiles
    output_file:                    str, the filename to use for saving the reconstructed map (full path)
    dataset.json:                   str, the full path to the dataset.json file produced during the tiling operation
                                    It is only used for retrieving the field `original_input_size`, a size tuple
                                    corresponding to the original size of the full image (i.e. pre-tiling) before any
                                    padding operations that took place
    colourize:                      boolean, if true slum pixels (1s) will be multipled by the value of 255
                                    so that they are visible when opened with image viewers 
ARGS:
    -c [--config_file]:             str, full path to the configuration file [default: \'reconstruct.yml\']
USAGE:
    >>> python3 reconstruct_map_from_tiles .py -c /path/to/reconstuct.yml
'''

import sys
import os
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
import yaml
import json
import click
from pathlib import Path
import pdb
try:
    from slumworldML.src.base_tiler import ImageTiler
except Exception as Error:
    try:
        from src.base_tiler import ImageTiler
    except Exception as Error2:
        from ..src.base_tiler import ImageTiler


@click.command()
@click.option('-c', '--config_file', default=None, help='Path to the configuration file. If None supply command line arguments')
@click.option('--dataset_json', default=None, help='Full path to the dataset.csv file.')
@click.option('--tile_folder_path', default=None, help='Full Path to the image_tiles that will be used for reconstruction.')
@click.option('--reconstructed_map_filepath', default=None, help= 'Full path and name for the reconstructed map file.')
@click.option('--colourize', default=False, help='Colourize the output for viewing with image viewers.')
@click.help_option('-h', '--help')
def main(config_file, dataset_json, tile_folder_path,  reconstructed_map_filepath, colourize):
    if config_file:
        if os.path.exists(config_file):
            with open(config_file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            with open(cfg['dataset_json'], 'r') as fin:
                params = json.load(fin)
            tile_folder_path = str(Path(cfg['image_dir']).absolute())
            output_filename = str(Path(cfg['output_file']).absolute())
            original_image_size = params['original_input_size']
            ImageTiler.reconstruct_image(   tile_folder_path=tile_folder_path, 
                                            output_filename=output_filename,
                                            target_size=original_image_size,
                                            colourize=cfg['colourize']
                                            )
            sys.exit(0)
        else:
            print("Could not find supplied configuration file.\n", Error)
            sys.exit(1)
    else:
        with open(dataset_json, 'r') as fin:
            params = json.load(fin)
        original_image_size = params['original_input_size']
        ImageTiler.reconstruct_image(   tile_folder_path=tile_folder_path, 
                                        output_filename=reconstructed_map_filepath,
                                        target_size=original_image_size,
                                        colourize=colourize
                                        )
        sys.exit(0)

if __name__ == "__main__":

    main()