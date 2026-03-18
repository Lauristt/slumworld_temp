import os
import sys
import numpy as np
import shutil
import cv2
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import imageio
import click
import yaml
from tqdm import tqdm
from multiprocessing import Pool
file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(file_dir)
sys.path.insert(0, parent_dir)
try:
    from .src.utilities import normalise_2D, crf
except Exception as Error:
    try:
        from utilities import normalise_2D, crf
    except Exception as Error2:
        try:
            from ..src.utilities import normalise_2D, crf
        except Exception as Error3:
            try:
               from src.utilities import normalise_2D, crf
            except Exception as Error4:
                from slumworldML.src.utilities import normalise_2D, crf
                print(Error1,Error2,Error3,Error4)

def worker(params):
    image_i, image_path, results_path, crf_results_path, \
                    kernel_size, compat, colour_kernel_size, colour_compat, n_steps = params
    if image_i.endswith('png') or image_i.endswith('jpg'):
        # print(image_i)
        try:
            X = cv2.imread(os.path.join(image_path, image_i))
            y = cv2.imread(os.path.join(results_path, image_i)) *255
        except Exception as Error:
            print(Error)
            print(30*"#")
            print(f"{os.path.join(image_path,image_i)}")
            print(f"{os.path.join(results_path,image_i)}")
            print(30*"#")

        if y.max() == 0:
            shutil.copyfile(os.path.join(results_path,image_i),
                        os.path.join(crf_results_path,image_i)
                        )
            return
        output_image = os.path.join(crf_results_path, image_i)
        try:
            crfimage = crf(original_image=X,
                        annotated_image=y, 
                        output_image=output_image,
                        n_steps=n_steps,
                        kernel_size=kernel_size, compat=compat,
                        colour_kernel_size=colour_kernel_size, colour_compat=colour_compat
                        )
        except Exception as Error:
            print(10*"#"+" Erro in crf processing "+10*"#")
            print(Error)
            print(f"{os.path.join(results_path,image_i)}")
            print(f"{os.path.join(image_path,image_i)}")
            print(30*"#")

@click.command()
@click.option('--config', '-c', default=None, help='Path to the configuration file')
@click.option('--results_path', default='/home/qigong/projects/slumworld/output/WB_models/PCMC/ensemble/evaluation_1024_ensemble9models/results', 
                help='Path to the NN results')
@click.option('--image_path', default='/home/qigong/projects/slumworld/data/tiled/PCMC/PAN/version002_1024/tiled_input', 
                help='Path to the images')
@click.option('--crf_results_path', default='/home/qigong/projects/slumworld/output/WB_models/PCMC/ensemble/evaluation_1024_ensemble9models/crf_resultsK10C3', 
                help='Path to the CRF results')
@click.option('--kernel_size', default=10, help='Kernel size')
@click.option('--compat', default=3, help='Compat')
@click.option('--colour_kernel_size', default=10, help='Colour kernel size')
@click.option('--colour_compat', default=3, help='Colour compat')
@click.option('--n_steps', default=10, help='Number of steps')
@click.option('--n_processes', default=10, help='Number of parallel processes')
@click.help_option('-h', '--help')
def process_images(config, results_path, image_path, crf_results_path, kernel_size, compat, 
                    colour_kernel_size, colour_compat, n_steps, n_processes):
    if config:
        if os.path.exists(config):
            with open(config, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
                results_path = cfg['results_path']
                image_path = cfg['image_path']
                crf_results_path = cfg['crf_results_path']
                kernel_size = cfg['kernel_size']
                compat = cfg['compat']
                colour_kernel_size = cfg['colour_kernel_size']
                colour_compat = cfg['colour_compat']
                n_steps = cfg['n_steps']
                n_processes = cfg['n_processes']
        else:
            print("Configuration file not found")
            return

    images2process = sorted([f for f in os.listdir(results_path) if (f.endswith(".png") or f.endswith(".jpg"))])
    print(f"\n\n######### Processing {len(images2process)} images #########")
    print("Checking if all images exist...")
    for file in images2process:
        if not os.path.isfile(os.path.join(results_path, file)):
            print(f"File {os.path.join(results_path, file)} does not exist! Check your configuration file!")
            return 1
    for file in images2process:
        if not os.path.isfile(os.path.join(image_path, file)):
            print(f"File {os.path.join(image_path, file)} does not exist! Check your configuration file!")
            return 1
    print("All files exist, proceeding to processing...")
    print(41*"#"+"\n\n")
    os.makedirs(crf_results_path, exist_ok=True)

        # Define pool
    with Pool(n_processes) as pool:
        params = [(image, image_path, results_path, crf_results_path, 
                    kernel_size, compat, colour_kernel_size, colour_compat, n_steps) for image in images2process] 
        list(tqdm(pool.imap(worker, params), total=len(images2process)))


if __name__ == '__main__':
    process_images()