''' A unified script to post process the model predictions (a collection of tiles) with a Conditional Random Field (CRF)
It consumes a configuration file 'crf_postprocess_pipeline.yml' and performs the following steps:
1. CRF processing of tiles
2. Evaluation vs a collection of true labels
3. Glueing the new prediction tiles together to a single 'reconstructed_map.png' file
4. Overlay the predicted tiles on the the true labels and and the underline raw satellite image-> outputs a png file
5. Generate a shapefile for viewing the results in Google earth or any GIS software
'''
import sys
import os
import click
import subprocess
import yaml


# Yuting Modified, Get the directory where the current script is located
# This assumes the script is run as a file, not just an imported module in a complex way
script_dir = os.path.dirname(os.path.abspath(__file__))
# --- End of added section ---

@click.command()
@click.option('--config', '-c', default=None, help='Path to the configuration file')
@click.help_option('-h', '--help')

def run(config):
    with open(config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Run scripts with parameters sequentially
    if cfg['tiled_labels_path'] is not None: # we have labels hence we run the evaluation part
        scripts = { 
                'crf_postprocess_parallel.py':[
                        '--results_path',str(cfg['results_path']),
                        '--image_path', str(cfg['tiled_image_path']),
                        '--crf_results_path', str(cfg['crf_results_path']),
                        '--kernel_size', str(cfg['kernel_size']),
                        '--compat', str(cfg['compat']),
                        '--colour_kernel_size', str(cfg['colour_kernel_size']),
                        '--colour_compat', str(cfg['colour_compat']),
                        '--n_steps', str(cfg['n_steps']),
                        '--n_processes', str(cfg['n_processes'])
                        ],
                'evaluate_from_folders.py':[
                        '--true_path', str(cfg['tiled_labels_path']),
                        '--pred_path', str(cfg['crf_results_path']),
                        '--threshold', str(cfg['threshold']),
                        '--results_file', os.path.join(str(cfg['crf_results_path']),
                                                        str(cfg['crf_evaluation_file'])),
                        '--processes', str(cfg['n_processes'])
                        ],
                'reconstruct_map_from_tiles.py':[
                        '--dataset_json', str(cfg['dataset_json']),
                        '--tile_folder_path', str(cfg['crf_results_path']),
                        '--reconstructed_map_filepath', os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['reconstructed_map_filename'])),
                        '--colourize', str(cfg['colourize']),
                        ],
                'overlay_map_on_image.py':[
                        '--satellite_img_file', str(cfg['satellite_img_file']), 
                        '--pred_slums_img_file', os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['reconstructed_map_filename'])), 
                        '--mask_file', str(cfg['mask_file']), 
                        '--true_slums_img_file', str(cfg['true_slums_img_file']), 
                        '--output_file', os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['map_overlay_filename'])), 
                        '--transparency', str(cfg['transparency'])
                        ],
                'generate_shapefile.py':[
                        '--input_image_path', str(cfg['satellite_img_file']),
                        '--auxilliary_files_folder', str(cfg['auxilliary_files_folder']),
                        '--output_folder', str(cfg['shapefile_location']),
                        '--shapefile_name', str(cfg['shapefile_name']),
                        '--reconstructed_map_file',  os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['reconstructed_map_filename'])),
                        '--crop', str(cfg['crop']),
                        '--epsg_code', str(cfg['epsg_code']), 
                        ]}
    else: # we do not have labels, hence there is no evaluation step
        scripts = { 
                'crf_postprocess_parallel.py':[
                        '--results_path',str(cfg['results_path']),
                        '--image_path', str(cfg['tiled_image_path']),
                        '--crf_results_path', str(cfg['crf_results_path']),
                        '--kernel_size', str(cfg['kernel_size']),
                        '--compat', str(cfg['compat']),
                        '--colour_kernel_size', str(cfg['colour_kernel_size']),
                        '--colour_compat', str(cfg['colour_compat']),
                        '--n_steps', str(cfg['n_steps']),
                        '--n_processes', str(cfg['n_processes'])
                        ],
                'reconstruct_map_from_tiles.py':[
                        '--dataset_json', str(cfg['dataset_json']),
                        '--tile_folder_path', str(cfg['crf_results_path']),
                        '--reconstructed_map_filepath', os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['reconstructed_map_filename'])),
                        '--colourize', str(cfg['colourize']),
                        ],
                'overlay_map_on_image.py':[
                        '--satellite_img_file', str(cfg['satellite_img_file']), 
                        '--pred_slums_img_file', os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['reconstructed_map_filename'])), 
                        '--mask_file', str(cfg['mask_file']), 
                        '--true_slums_img_file', str(cfg['true_slums_img_file']), 
                        '--output_file', os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['map_overlay_filename'])), 
                        '--transparency', str(cfg['transparency'])
                        ],
                'generate_shapefile.py':[
                        '--input_image_path', str(cfg['satellite_img_file']),
                        '--auxilliary_files_folder', str(cfg['auxilliary_files_folder']),
                        '--output_folder', str(cfg['shapefile_location']),
                        '--shapefile_name', str(cfg['shapefile_name']),
                        '--reconstructed_map_file',  os.path.join(str(cfg['crf_results_path']),
                                                                str(cfg['reconstructed_map_filename'])),
                        '--crop', str(cfg['crop']),
                        '--epsg_code', str(cfg['epsg_code']), 
                        ]}
    for script, params in scripts.items():
        # print(f"{script} {' '.join(params)}")
        # subprocess.call(['python', script, *params])
        ## If recover plz comment out the above two lines of code, Yuting Modified, to use the absolute path
        print(f"{script} {' '.join(params)}")
        # --- Modify this line ---
        # Construct the full path to the script
        script_path = os.path.join(script_dir, script)
        # Call subprocess.call with the full path
        subprocess.call(['python', script_path, *params])
        # --- End of modified line ---

if __name__ == '__main__':
    # Read configuration from YML file
    run()