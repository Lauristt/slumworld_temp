import os
from PIL import Image
import skimage
import click
from tqdm import tqdm

@click.command()
@click.option('-i', '--input_folder',  default='/home/qigong/projects/slumworld/data/tiled/Bobo/version002_1024/tiled_input_RGB/', 
                help='RGB image folder.')
@click.option('-o', '--output_folder', default='/home/qigong/projects/slumworld/data/tiled/Bobo/version002_1024/tiled_input/', 
                help='Where to save the grayscale images. Will be created if non existent.')
@click.help_option('-h', '--help')
def convert(input_folder, output_folder):
    '''this functions conerts all rgb images in the input path to grayscale images and saves them in the output path'''
    image_list = os.listdir(input_folder)
    if len(image_list) == 0:
        print("No files found in input directory")
        return 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image in tqdm(image_list):
        if not image.startswith('.'):
            img = Image.open(os.path.join(input_folder, image)).convert('L').convert('RGB')
            # img = skimage.color.gray2rgb(img)
            img.save(os.path.join(output_folder, image))
    print("Operation completed succesfully.")

if __name__ == "__main__":
    convert()

