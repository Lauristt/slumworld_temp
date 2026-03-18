''' Generates shapefiles from the results of map prediction on a satellite image. 
Requires a configuration file which will include information aboutthe raw satellite image, the predicted
    slum map image, as well as the (.png.aux.xml, .png.ovr, .png.xml, and .pgw) auxilliary files produced during image generation. 
    Assumes each auxiliary file has the same basename (i.e. excluding the suffix).

CONFIG file (yml) fields:
        input_image_path:                   str, path to the raw satellite image we are predicting on (input_x)
        output_shapefile_path:              str, filename and path to the folder where the shapefile will be saved
        auxilliary_files_folder_path:       str, path to the folder holding the auxilliary files produced during image generation. 
                                            The folder should contain the following 4 files - each having the same basename (prefix) as the input_image:
                                            *.png.aux.xml, *.png.ovr, *.png.xml, *.pgw 
        reconstructed_map_file_path:        str, path to the reconstructed map png file
        pan_flag:                           boolean, True: 1 channel, False: 3 channels
    
USAGE:
    >>> python3 generate_shapefile.py -c ../config/generate_shapefile.yml
'''
import sys
import os
import yaml
sys.path.append("..")
__package__ = os.path.dirname(sys.path[0])
from pathlib import Path
import click
try:
    from slumworldML.src.inspector import generate_shapefiles
except Exception as Error:
    try:
        from src.inspector import generate_shapefiles
    except Exception as Error2:
        from ..src.inspector import generate_shapefiles

@click.command()
@click.option('-c', '--config_file', default=None, help='Path to the configuration file. If None supply command line arguments')
@click.option('--input_image_path', default=None, help='Path to the raw satellite image we are predicting on (input_x).')
@click.option('--auxilliary_files_folder', default=None, help='Path to the folder holding the auxilliary files produced during image generation.')
@click.option('--output_folder', default=None, help= 'Full path to the folder that the shapefile (a folder itself) will be created in.')
@click.option('--shapefile_name', default=None, help='Filename for the produced shapefile.')
@click.option('--reconstructed_map_file', default=None, help='Path to the reconstructed map png file')
@click.option('--crop', default=True, help='if set to false no cropping will be applied to the image [default: true]')
@click.option('--epsg_code', default=None, help='EPSG code for city [mumbai/PCMC: 32643, burkina fasos:32630], for others google')
@click.help_option('-h', '--help')
def run(config_file, input_image_path, auxilliary_files_folder, output_folder,
        shapefile_name, reconstructed_map_file, crop, epsg_code, 
        produce_png_overlay=False):
    if config_file:
        if os.path.exists(config_file):
            with open(config_file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            generate_shapefiles(input_image_path=cfg['input_image_path'],
                                auxilliary_files_folder=cfg['auxilliary_files_folder'],
                                output_folder=cfg['output_folder'],
                                shapefile_name=cfg['shapefile_name'],
                                reconstructed_map_file=cfg['reconstructed_map_file'],
                                crop=cfg['crop'],
                                epsg_code=cfg['epsg_code'], 
                                produce_png_overlay=cfg['produce_png_overlay']
                                )
            sys.exit(0)
        else:
            print("Config file not found:", Error)
            sys.exit(1)
    try:
        generate_shapefiles(input_image_path=input_image_path,
                            auxilliary_files_folder=auxilliary_files_folder,
                            output_folder=output_folder,
                            shapefile_name=shapefile_name,
                            reconstructed_map_file=reconstructed_map_file,
                            crop=crop,
                            epsg_code=epsg_code, 
                            produce_png_overlay=produce_png_overlay
                                )
        sys.exit(0)
    except Exception as Error:
        print("Error during shapefile generation!\nError log:", Error)
        sys.exit(2)
if __name__ == '__main__':

    run()