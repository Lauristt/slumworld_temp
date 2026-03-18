import sys
import os
import shutil
from pathlib import Path
import math
from collections import defaultdict
import numpy as np
import pickle
import pdb
import datetime
import rasterio as rio
from rasterio.features import shapes
import geopandas as gpd
import imageio
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import time
try:
    from slumworldML.src.base_tiler import ImageTiler
    from slumworldML.src.utilities import create_save_directory
except Exception as Error:
    try:
        from base_tiler import ImageTiler
        from utilities import create_save_directory
    except Exception as Error2:
        try:
            from src.base_tiler import ImageTiler
            from src.utilities import create_save_directory
        except Exception as Error3:
            from .src.base_tiler import ImageTiler
            from .src.utilities import create_save_directory

def compress(input_path, output_path, filename = None, compression_ratio=0.1) :
    '''Generates smaller version of an image.
    Args:
        input_path:                         str, path to the folder containing the image
        output_path:                        str, path to the output folder where the compressed image will be saved
        filename:                           str, file name of image to be compressed
        compression_ratio:                  float, ratio of compressed size to original size [default: 0.1]
    Returns:
        nothing, saves the produced .png file
    Usage:
        >>> input_path = '/home/user/slumworld/data/raw/MD_MUL_97_Brenda/'
        >>> output_path = '.'
        >>> compress(input_path, output_path, 'input_x.png', 0.1)
    '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not filename.endswith('.png'): filename += '.png'
    image = Image.open(input_path/filename)
    resized = image.resize((int(image.width*compression_ratio), int(image.height*compression_ratio)))
    resized.save(output_path/filename)


def overlay(satellite_img_file, pred_slums_img_file, output_file, mask_file=None, true_slums_img_file=None, transparency=0.5, prediction_mode='binary'):
    '''Overlays predicted slums and possibly true slums (if existent) onto the original satellite image, highlighting the aread used for calculations (i.e. the non-masked area).
    Colour coding is: (i). Predicted Slums: RED, (ii). True Slums: BLUE, (iii). Predicted Slums = True Slums: MAGENTA, (iv). Area used for calculations: PALE GREEN
    Args:
        satellite_img_file:                 str, full path (and name) to the satellite image used for prediction (i.e. input_x)
        pred_slums_img_file:                str, full path (and name) to the file containing the predicted slum map (i.e. a single file reconstructed from the tiles)
        output_file:                        str, full path (and name) of the output file that will be saved
        mask_file:                          str, full path (and name) to the mask image, if used, (i.e. input_z) [default: None]
        true_slums_img_file:                str, full path (and name) to the file containing the true slums (i.e. the labels, input_y) [default: None]
        transparency:                       float (0,1): transparency of slum coloring [default: 0.5]
        prediction_mode:                    str ['binary', 'other'] indicates whether the model predicts yes/no slum locations 
                                            or distance to the nearest slum [default: 'binary']
    Returns:
        nothing, saves the produced .png file
    Usage:
        >>> overlay(input_path, output_path, input_x.png, reconstructed_predictions.png, overlayed_image.png, 
                    mask_file = input_z.png, transparency = 0.8, prediction_mode = 'binary')
    '''
    GREEN_CHANNEL_ADJUSTMENT_FACTOR = 0.3
    if true_slums_img_file == 'None': true_slums_img_file = None
    if mask_file == 'None': mask_file = None
    if not satellite_img_file.endswith('.png'): satellite_img_file += '.png'
    if not pred_slums_img_file.endswith('.png'): pred_slums_img_file += '.png'
    if not output_file.endswith('.png'): output_file += '.png'
    if (mask_file is not None) and (not mask_file.endswith('.png')): mask_file += '.png'
    print(f"mask_file: {mask_file}\ntrue_slums_img_file: {true_slums_img_file}")
    slums_img = Image.open(pred_slums_img_file)
    slums_img_arr = np.asarray(slums_img, dtype='uint8')
    if (len(slums_img_arr.shape) == 3) & (slums_img_arr.shape[-2] > 1):
        slums_img_arr = slums_img_arr[:,:,0]
    out_img = Image.open(satellite_img_file)
    out_img_arr = np.array(out_img, dtype='uint8')# reconstruct error
    if len(out_img_arr.shape) == 2 or out_img_arr.shape[-1] == 1 :   
        # this is a PAN image, we have to transform it to RGB
        out_img_arr = np.tile(out_img_arr[:,:,None], [1,1,3])
    if prediction_mode == 'binary' :
        SLUM_VALUE = slums_img_arr.max()
        if mask_file is None:
            # set predicted slums to RED
            out_img_arr[(SLUM_VALUE == slums_img_arr), 0] = 255 * (1-transparency) + out_img_arr[(SLUM_VALUE == slums_img_arr), 0] * transparency
            out_img_arr[(SLUM_VALUE == slums_img_arr), 1] = 0 * (1-transparency) + out_img_arr[(SLUM_VALUE == slums_img_arr), 1] * transparency
            out_img_arr[(SLUM_VALUE == slums_img_arr), 2] = 0 * (1-transparency) + out_img_arr[(SLUM_VALUE == slums_img_arr), 2] * transparency
            # set mask background to GREEN (reduce brightness by GREEN_CHANNEL_ADJUSTMENT_FACTOR)
            # out_img_arr[0 == slums_img_arr, 0] = 0 * (1-transparency) + out_img_arr[0 == slums_img_arr, 0] * transparency
            # out_img_arr[0 == slums_img_arr, 1] = GREEN_CHANNEL_ADJUSTMENT_FACTOR * 255 * (1-transparency) + out_img_arr[0 == slums_img_arr, 1] * transparency
            # out_img_arr[0 == slums_img_arr, 2] = 0 * (1-transparency) + out_img_arr[0== slums_img_arr, 2] * transparency
            if true_slums_img_file is not None:
                true_slums_img = Image.open(true_slums_img_file)
                true_slums_img_arr = np.asarray(true_slums_img, dtype='uint8')
                if true_slums_img_arr.max() > 1:
                    true_slums_img_arr = (true_slums_img_arr < 64).astype('uint8') # convert to binary 
                # set true slums to BLUE
                out_img_arr[(0 == true_slums_img_arr), 0] = 0 * (1-transparency) +\
                                                   out_img_arr[(0 == true_slums_img_arr), 0] * transparency
                out_img_arr[(0 == true_slums_img_arr), 1] = 0 * (1-transparency) +\
                                                   out_img_arr[(0 == true_slums_img_arr), 1] * transparency
                out_img_arr[(0 == true_slums_img_arr), 2] = 255 * (1-transparency) +\
                                                    out_img_arr[(0 == true_slums_img_arr), 2] * transparency
        else:
            mask_img = Image.open(mask_file)
            mask_img_arr = np.asarray(mask_img, dtype='uint8')
            MASK_VALUE = mask_img_arr.max()
            # set predicted slums to RED
            out_img_arr[(SLUM_VALUE == slums_img_arr) & (mask_img_arr > 0), 0] = 255 * (1-transparency) + out_img_arr[(SLUM_VALUE == slums_img_arr) & (mask_img_arr > 0), 0] * transparency
            out_img_arr[(SLUM_VALUE == slums_img_arr) & (mask_img_arr > 0), 1] = 0 * (1-transparency) + out_img_arr[(SLUM_VALUE == slums_img_arr) & (mask_img_arr > 0), 1] * transparency
            out_img_arr[(SLUM_VALUE == slums_img_arr) & (mask_img_arr > 0), 2] = 0 * (1-transparency) + out_img_arr[(SLUM_VALUE == slums_img_arr) & (mask_img_arr > 0), 2] * transparency
            # set mask background to GREEN (reduce brightness by GREEN_CHANNEL_ADJUSTMENT_FACTOR)
            # out_img_arr[(0 == slums_img_arr) & (mask_img_arr > 0), 0] = 0 * (1-transparency) + \
            #                                 out_img_arr[(0 == slums_img_arr) & (mask_img_arr > 0), 0] * transparency
            # out_img_arr[(0 == slums_img_arr) & (mask_img_arr > 0), 1] = GREEN_CHANNEL_ADJUSTMENT_FACTOR * 255 * (1-transparency) + \
            #                                 out_img_arr[(0 == slums_img_arr) & (mask_img_arr > 0), 1] * transparency
            # out_img_arr[(0 == slums_img_arr) & (mask_img_arr > 0), 2] = 0 * (1-transparency) + \
            #                                 out_img_arr[(0 == slums_img_arr) & (mask_img_arr > 0), 2] * transparency
            if true_slums_img_file is not None:
                true_slums_img = Image.open(true_slums_img_file)
                true_slums_img_arr = np.asarray(true_slums_img, dtype='uint8')
                if true_slums_img_arr.max() > 1:
                    true_slums_img_arr = (true_slums_img_arr < 64).astype('uint8') # convert to binary 
                # set true slums to BLUE
                out_img_arr[(0 == true_slums_img_arr) & (mask_img_arr > 0), 0] = 0 * (1-transparency) +\
                                                   out_img_arr[(0 == true_slums_img_arr) & (mask_img_arr > 0), 0] * transparency
                out_img_arr[(0 == true_slums_img_arr) & (mask_img_arr > 0), 1] = 0 * (1-transparency) +\
                                                   out_img_arr[(0 == true_slums_img_arr) & (mask_img_arr > 0), 1] * transparency
                out_img_arr[(0 == true_slums_img_arr) & (mask_img_arr > 0), 2] = 255 * (1-transparency) +\
                                                    out_img_arr[(0 == true_slums_img_arr) & (mask_img_arr > 0), 2] * transparency
    else:
        if mask_file is None:
            # set predicted slums to RED
            out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127), 0] = 255 * (1-transparency) + out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127), 0] * transparency
            out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127), 1] = 0 * (1-transparency) + out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127), 1] * transparency
            out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127), 2] = 0 * (1-transparency) + out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127), 2] * transparency
            # set mask background to GREEN (reduce brightness by GREEN_CHANNEL_ADJUSTMENT_FACTOR)
            # out_img_arr[127 < slums_img_arr, 0] = 0 * (1-transparency) + out_img_arr[127 < slums_img_arr, 0] * transparency
            # out_img_arr[127 < slums_img_arr, 1] = 255 * (1-transparency) + out_img_arr[127 < slums_img_arr, 1] * transparency
            # out_img_arr[127 < slums_img_arr, 2] = 0 * (1-transparency) + out_img_arr[127 < slums_img_arr, 2] * transparency
            if true_slums_img_file is not None:
                true_slums_img = Image.open(true_slums_img_file)
                true_slums_img_arr = np.asarray(true_slums_img, dtype='uint8')
                # set true slums to BLUE
                out_img_arr[127 < true_slums_img_arr, 0] = 0 * (1-transparency) + out_img_arr[127 < true_slums_img_arr, 0] * transparency
                out_img_arr[127 < true_slums_img_arr, 1] = 0 * (1-transparency) + out_img_arr[127 < true_slums_img_arr, 1] * transparency
                out_img_arr[127 < true_slums_img_arr, 2] = 255 * (1-transparency) + out_img_arr[127 < true_slums_img_arr, 2] * transparency
        else:
            mask_img = Image.open(mask_file)
            mask_img_arr = np.asarray(mask_img, dtype='uint8')
            # set predicted slums to RED
            out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127) & (mask_img_arr < 127), 0] = 255 * (1-transparency) + out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127) & (mask_img_arr < 127), 0] * transparency
            out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127) & (mask_img_arr < 127), 1] = 0 * (1-transparency) + out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127) & (mask_img_arr < 127), 1] * transparency
            out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127) & (mask_img_arr < 127), 2] = 0 * (1-transparency) + out_img_arr[(64 <= slums_img_arr) & (slums_img_arr <= 127) & (mask_img_arr < 127), 2] * transparency
            # set mask background to GREEN (reduce brightness by GREEN_CHANNEL_ADJUSTMENT_FACTOR)
            # out_img_arr[(127 < slums_img_arr) & (mask_img_arr < 127), 0] = 0 * (1-transparency) + out_img_arr[(127 < slums_img_arr) & (mask_img_arr < 127), 0] * transparency                
            # out_img_arr[(127 < slums_img_arr) & (mask_img_arr < 127), 1] = 255 * (1-transparency) + out_img_arr[(127 < slums_img_arr) & (mask_img_arr < 127), 1] * transparency
            # out_img_arr[(127 < slums_img_arr) & (mask_img_arr < 127), 2] = 0 * (1-transparency) + out_img_arr[(127 < slums_img_arr) & (mask_img_arr < 127), 2] * transparency
            if true_slums_img_file is not None:
                true_slums_img = Image.open(true_slums_img_file)
                true_slums_img_arr = np.asarray(true_slums_img, dtype='uint8')
                # set true slums to BLUE
                out_img_arr[(127 < true_slums_img_arr) & (mask_img_arr < 127), 0] = 0 * (1-transparency) + out_img_arr[(127 < true_slums_img_arr) & (mask_img_arr < 127), 0] * transparency                
                out_img_arr[(127 < true_slums_img_arr) & (mask_img_arr < 127), 1] = 0 * (1-transparency) + out_img_arr[(127 < true_slums_img_arr) & (mask_img_arr < 127), 1] * transparency
                out_img_arr[(127 < true_slums_img_arr) & (mask_img_arr < 127), 2] = 255 * (1-transparency) + out_img_arr[(127 < true_slums_img_arr) & (mask_img_arr < 127), 2] * transparency

    output = Image.fromarray(out_img_arr.astype(np.uint8))
    create_save_directory(os.path.dirname(output_file))
    output.save(output_file)


def animate(input_path, image_list, output_filename, fps=2, add_img_labels=True, font_file=None):
    '''Generates a .gif that goes through all images in image_list with option to label each image.
    Args:
        input_path:                         str, path to the folder containing the images in the animation
        image_list:                         list of str, contains name of each image file used in the animation
        output_filename:                    str, full path and name of the file to be saved
        fps:                                int, animation speed in frames per second
        add_img_labels                      boolean, whether to label each image
    Returns:
        nothing, saves the produced .gif file
    Usage:
        >>> animate(input_path, output_path, image_list, output_filename, fps, add_img_labels)
    '''
    #font = ImageFont.truetype(font_file, 72)
    # font = ImageFont.truetype(font_file, 72)
    font = ImageFont.load_default()
    font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
    images = []
    input_path = Path(input_path)
    frame = 0
    for filename in image_list:
        im = Image.open(input_path/filename).convert("RGBA")
        if add_img_labels:
            txt = Image.new("RGBA", im.size, (255,255,255,0))
            draw = ImageDraw.Draw(txt)
            if 'epoch' in filename:
                text = 'epoch:' + format(int(filename.split('epoch_')[1].split('_')[0]),'04d')
            else:
                text = 'frame:' + format(frame,'04d')
            draw.text((im.size[0]//3,im.size[1]-20), text, (255,255,255), align='center', font=font)
            frame += 1
            composite = Image.alpha_composite(im, txt)
            images.append(composite)
        else:
            images.append(im)
    try:
        if output_filename.endswith('.png') or output_filename.endswith('.jpg'):
            output_filename = output_filename[:-4] + '.gif'
        if not output_filename.endswith('.gif'):
            output_filename += '.gif'
        imageio.mimsave(output_filename, images, fps=fps, format='gif')
        print("Animation file saved to:", output_filename)
    except Exception as Error:
        print("Error while saving animation:", Error)

def generate_random_string_from_time():
    current_time = datetime.datetime.now()
    random_string = current_time.strftime("%Y%m%d%H%M%S%f")
    return random_string

def generate_shapefiles(input_image_path, output_folder, auxilliary_files_folder, reconstructed_map_file, 
                        shapefile_name='predicted_slums_shapefile', produce_png_overlay=False, mode = 'binary', 
                        crop=True, epsg_code=32634):
    '''Generates shapefiles from the results of predicting on a test input. Requires the test .png image as well
    as .png.aux.xml, .png.ovr, .png.xml, and .pgw auxilliary files. Assumes each auxilliary file has the same name
    excluding suffix.
    Args:
        input_image_path:                   str, path to the image we are predicting on ('input_x.png')
        output_folder:                      str, path to the folder where the shapefiles will be saved
        auxilliary_files_folder:            str, path to the folder holding the auxilliary files produced during image generation. 
                                            The folder should contain the following 4 files - each having the same basename (prefix) as the input_image:
                                            *.png.aux.xml, *.png.ovr, *.png.xml, *.pgw 
        reconstructed_map_file:             str, path to the reconstructed map png file
        shapefile_name:                     str, basename of the produced shapefile [default: 'predicted_slums_shapefile']
        produce_png_overlay:                boolean, if true a png file with the shapfile map overlayed on the satellite image will be produced in the output folder
        mode:                               string, 'binary' indicates that we are just predicting yes/no for each pixel
        crop:                               boolean, if set to False no cropping will be applied to the shapefile [default: True]
        epsg_code:                          int, the epsg_code for the areas (default: 32636[Mumbai,PCMC], 32630[Bobo,Ouagadougou])
    Returns:
        nothing, saves the produced .shp and auxilliary files
    Usage:
        >>> input_image_path = '/home/user/data/raw/Mumbai/MS/inputs/input_x.png'
        >>> auxilliary_files_folder = '/home/user/data/raw/Mumbai/MS/aux_files/'
        >>> output_folder = '/home/user/output/experiment/deeplabv3/lightning_logs/version_002/'
        >>> reconstructed_map_file = '/home/user/output/experiment/deeplabv3/lightning_logs/version_002/reconstructed_map.png'
        >>> shapefile_name='predicted_slums_shapefile'
        >>> generate_shapefiles(input_image_path, output_folder, auxilliary_files_folder, reconstructed_map_file, shapefile_name,
                                produce_png_overlay=False, mode = 'binary', crop=True)
    '''
    # ensure paths are strings (if passed arguments are PosixPaths)
    auxilliary_files_folder = str(auxilliary_files_folder)
    input_image_path = str(input_image_path)
    output_folder = str(output_folder)
    shapefile_name = str(shapefile_name)
    # create a tmp path
    temp_path = os.path.join(os.path.dirname(output_folder), generate_random_string_from_time())
    os.makedirs(temp_path, exist_ok=True)
    temp_path_1 = os.path.abspath(os.path.join(temp_path, generate_random_string_from_time()))
    ### copy auxilliary files into input_image_path
    input_dir = os.path.dirname(os.path.abspath(input_image_path))
    input_files = os.listdir(input_dir)
    aux_files = os.listdir(auxilliary_files_folder)
    [shutil.copyfile(os.path.join(auxilliary_files_folder, file), os.path.join(input_dir, file)) 
                    for file in aux_files]

    img = Image.open(reconstructed_map_file)
    if len(np.array(img).shape) == 3:
         distance_estimated = np.array(img)[:,:,0]
    else:
        distance_estimated = np.array(img)
    if mode == 'binary':
        distance_estimated[distance_estimated>0.5] = 127 

    sat_img_raw = Image.open(input_image_path)
    n_col, n_row = sat_img_raw.size
    width  = np.int_(math.floor(n_col / 16.) * 16)
    height = np.int_(math.floor(n_row / 16.) * 16)
    sat_img_uncrop = np.asarray(sat_img_raw, dtype = 'float32')
    if len(sat_img_uncrop.shape) == 2:
        # pan image, only 2-dimensions, copy along new dimension to make 3 channel
        sat_img_uncrop = np.stack((sat_img_uncrop,)*3, axis=-1)

    # crop images 
    if crop:
        sat_img_unnorm = sat_img_uncrop[0:height,0:width,:]
        distance_estimated = distance_estimated[0:height,0:width]
    else:
        sat_img_unnorm = sat_img_uncrop
        distance_estimated = distance_estimated

    sat_img_unnorm[128 <= distance_estimated * 255, 0] = 255
    sat_img_unnorm[(127 < distance_estimated * 255) & (distance_estimated * 255 < 128), 2] = 255

    output_img = Image.fromarray(sat_img_unnorm.astype(np.uint8))
    output_img.save(input_image_path + 'k.png')

    # Generating an additional image for conversion to shapefile
    sat_img_unnorm[(127 < distance_estimated * 255) & (distance_estimated * 255 < 128), 0] = 127
    sat_img_unnorm[distance_estimated * 255 <= 127, 0] = 0

    output_img_shp = Image.fromarray(sat_img_unnorm.astype(np.uint8))

    inRaster = input_image_path + 'k_shp.png'

    output_img_shp.save(inRaster)

    inarray=rio.open(inRaster).read(1).astype('uint16')
    inarray[inarray!=255]=65535
    inarray[inarray==255]=1

    outarray = inarray[np.newaxis,:,:]

    in_meta = rio.open(input_image_path).meta.copy()
    in_meta.update({"driver": "GTiff",
                    'dtype': 'uint16',
                     "count": 1,
                     'width': width,
                     'height':height
                     })
    with rio.open(temp_path_1, 'w', **in_meta) as output_tif: 
        output_tif.write(outarray)
        
    mask = None
    with rio.Env():
        with rio.open(temp_path_1) as src:
            image = src.read(1) # first band
            results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                shapes(image, mask=mask, transform=src.transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(epsg=epsg_code)
    gpd_polygonized_raster[:-1].to_file(output_folder+'/'+shapefile_name)
    ### cleanup
    shutil.rmtree(temp_path)
    # we will not cleanup incase there is more than one instance running at the same time
    # [os.remove(os.path.join(input_dir, file)) for file in aux_files]
    if produce_png_overlay:
        [shutil.move(os.path.join(input_dir, file), os.path.join(output_folder, file)) for file in os.listdir(input_dir) if file not in input_files]
    else:
        [os.remove(os.path.join(input_dir, file)) for file in os.listdir(input_dir) if file not in input_files+aux_files]


def resynthesize(tile_folder_path, output_filename, target_size=None):
    '''Reconstructs an image from tiles.
    Args:
        tile_folder_path:                   str, path to the folder containing the tiles
        output_filename:                    str, name of the file which the output will be saved
        target_size:                        (int, int), optional. If an input is provided, the resulting
                                                image will be truncated (intended to remove any padding added during
                                                tiling, otherwise padding will be included.
    Returns:
        nothing, saves the resulting map.
    Usage:
        >>> resynthesize(tile_folder_path, output_filename, target_size)
    '''
    ImageTiler.reconstruct_image(tile_folder_path=tile_folder_path,
                            output_filename=output_filename,
                            target_size=target_size)
    print("Operation completed. Resulting map saved to:", output_filename)


def animate_progress_dir(prediction_samples_folder, animation_folder, fps=10.):
    '''Wrapper function to be used to animate prediction samples saved periodically during model training
    Assumes the following naming convention for saved samples \'Epoch_{epoch}_{x_cord}_{y_cord}.png\
    Args:
        predictions_samples_folder:         str, path to the folder containing the saved image tiles
        animation_folder:                   str, path to the output folder, i.e. where the animations will be saved
        fps:                                int, the animation speed in frames per second [default:10]
    Returns:
        nothing, saves the produced *.gif files
    Usage:
        >>> animate_progress_dir(my_prediction_samples_folder, my_animation_folder, fps=10.)
    '''
    try:
        files = os.listdir(prediction_samples_folder)
    except Exception as Error:
        print("Input folder could not be located:", Error)
        sys.exit(1)
    if not os.path.exists(animation_folder):
        os.makedirs(animation_folder, exist_ok=True)
    res = defaultdict(list)
    for f in files:
        res['_'.join(f.split('_')[2:])].append(f)
    res = { k:sorted(v) for k, v in res.items()}
    for file_i in res.keys():
        animate(prediction_samples_folder, animation_folder, res[file_i], file_i, fps=fps, add_img_labels=False)

### older versions of generate_shapefiles function
def generate_shapefiles_arcpy(input_image_path, auxilliary_files_folder_path, output_shapefile_path, reconstructed_map_file_path, pan_flag = False, mode = 'binary'):
    '''Generates shapefiles from the results of map prediction on a satellite image. Requires the raw satellite image, the predicted
    slum map image, as well as the (.png.aux.xml, .png.ovr, .png.xml, and .pgw) auxilliary files produced during image generation. 
    Assumes each auxilliary file has the same basename (i.e. excluding the suffix).
    Args:
        input_image_path:                   str, path to the raw satellite image we are predicting on (input_x)
        output_shapefile_path:              str, filename and path to the folder where the shapefile will be saved
        auxilliary_files_folder_path:       str, path to the folder holding the auxilliary files produced during image generation. 
                                            The folder should contain the following 4 files - each having the same basename (prefix) as the input_image:
                                            *.png.aux.xml, *.png.ovr, *.png.xml, *.pgw 
        reconstructed_map_file_path:        str, path to the reconstructed map png file
        pan_flag:                           boolean, True: 1 channel, False: 3 channels
        mode:                               string, 'binary' indicates that we are just predicting yes/no for each pixel
    Returns:
        nothing, saves the produced .shp and auxilliary files
    Usage:
        >>> generate_shapefiles(input_image_path, output_shapefile_path, temp_path_1, reconstructed_map, pan_flag, mode)
    '''
    import rasterio as rio
    import geopandas as gpd
    try:
        import arcpy
        from arcpy import env
        from arcpy.sa import RemapValue, Reclassify
    except Exception as Error:
        print("Problem with arcpy installation: Aborting ...\n", Error)
    ### copy auxilliary files into input_image_path
    input_dir = os.path.dirname(os.path.abspath(input_image_path))
    aux_files = os.listdir(auxilliary_files_folder_path)
    [shutil.copyfile(os.path.join(auxilliary_files_folder_path, file), os.path.join(input_dir, file)) 
                    for file in aux_files]
    ### create 3 tmp files 
    output_dir = os.path.dirname(os.path.abspath(input_image_path))
    temp_path_1 = os.path.join(output_dir, generate_random_string_from_time())
    temp_path_2 = os.path.join(output_dir, generate_random_string_from_time())
    temp_path_3 = os.path.join(output_dir, generate_random_string_from_time())
    
    img = Image.open(reconstructed_map_file_path)
    predicted_map = np.array(img)[:,:,0]
    if mode == 'binary':
        predicted_map *= 127 

    test_img_raw = Image.open(input_image_path + '.png')

    n_col, n_row = test_img_raw.size
    width  = np.int(math.floor(n_col / 16.) * 16)
    height = np.int(math.floor(n_row / 16.) * 16)

    test_img_uncrop = np.asarray(test_img_raw, dtype = 'float32')

    try: _, _, n_bands = test_img_uncrop.shape

    except:
        h_uncrop, w_uncrop = test_img_uncrop.shape
        n_bands            = 1
        test_img_uncrop    = test_img_uncrop.reshape((h_uncrop, w_uncrop, n_bands))

    test_img_unnorm = test_img_uncrop[0:height, 0:width, 0:n_bands]

    if pan_flag:
        test_img_unnormn = np.zeros((height, width, 3), dtype='uint8')
        test_img_unnormn[:,:,0] = test_img_unnorm[:,:,0]
        test_img_unnormn[:,:,1] = test_img_unnorm[:,:,0]
        test_img_unnormn[:,:,2] = test_img_unnorm[:,:,0]
        test_img_unnorm=test_img_unnormn

    test_img_unnorm[128 <= predicted_map * 255, 0] = 255
    test_img_unnorm[(127 < predicted_map * 255) & (predicted_map * 255 < 128), 2] = 255

    output_img = Image.fromarray(test_img_unnorm.astype(np.uint8))
    output_img.save(input_image_path + 'k.png')

    # Generating an additional image for conversion to shapefile
    test_img_unnorm[(127 < predicted_map * 255) & (predicted_map * 255 < 128), 0] = 127
    test_img_unnorm[predicted_map * 255 <= 127, 0] = 0

    output_img_shp = Image.fromarray(test_img_unnorm.astype(np.uint8))

    inRaster = input_image_path + 'k_shp.png'

    output_img_shp.save(inRaster)

    reclassField = "VALUE"
    remap = RemapValue([[0, "NODATA"], [255, 1], ["NODATA", "NODATA"]])
    outReclassify = Reclassify(inRaster, reclassField, remap)
    if arcpy.Exists(os.getcwd()):
        arcpy.Delete_management(os.getcwd())

    outReclassify.save(temp_path_1)

    in_meta = rio.open(input_image_path).meta.copy()
    temp_tif= rio.open(temp_path_1)
    in_meta.update({"driver": "GTiff",
                    'dtype': 'uint16',
                     "count": 1,
                     'width': temp_tif.read().shape[2]
                     })
    temp_tif=temp_tif.read()
    with rio.open(temp_path_2, 'w', **in_meta) as output_tif: 
        output_tif.write(temp_tif)

    arcpy.RasterToPolygon_conversion(temp_path_2, temp_path_3)
    
    slum_gdf = gpd.read_file(temp_path_3)
    slum_gdf[:-1].to_file(output_shapefile_path)
    ### cleanup
    [os.remove(os.path.join(input_dir, file)) for file in aux_files]
    [os.remove(file) for file in [temp_path_1, temp_path_2, temp_path_3] ]


def generate_shapefiles_arcpy_theano(input_image_path, output_path, temp_path, est_dist_path, pan_flag = False, mode = 'binary'):
    '''Generates shapefiles from the results of predicting on a test input using the theano model. Requires the test .png image as well
    as .png.aux.xml, .png.ovr, .png.xml, and .pgw auxilliary files. Assumes each auxilliary file has the same name
    excluding suffix.
    Args:
        input_image_path:                   str, path to the image we are predicting on
        output_path:                        str, path to the output folder where the shapefiles will be saved
        temp_path:                          str, path to an arbitrary folder used to store intermediate outputs
        est_dist_path:                      str, path to the pkl file containing the distance_estimated asarray
        pan_flag:                           boolean, True: 1 channel, False: 3 channels
        mode:                               string, 'binary' indicates that we are just predicting yes/no for each pixel
    Returns:
        nothing, saves the produced .shp and auxilliary files
    Usage:
        >>> generate_shapefiles(input_image_path, output_path, temp_path, est_dist_path, pan_flag, mode)
    '''

    try:
        import arcpy
        from arcpy import env
        from arcpy.sa import RemapValue, Reclassify
    except Exception as Error:
        print("Problem with arcpy installation: Aborting ...\n", Error)
    with open(est_dist_path + '.pkl', 'rb') as f:
        distance_estimated = pickle.load(f, encoding='latin1')
    if mode == 'binary':
        distance_estimated *= 127 

    test_img_raw = Image.open(input_image_path + '.png')

    n_col, n_row = test_img_raw.size
    width  = np.int(math.floor(n_col / 16.) * 16)
    height = np.int(math.floor(n_row / 16.) * 16)
    print(width)
    print(height)

    test_img_uncrop = np.asarray(test_img_raw, dtype = 'float32')

    try: _, _, n_bands = test_img_uncrop.shape

    except:
        h_uncrop, w_uncrop = test_img_uncrop.shape
        n_bands            = 1
        test_img_uncrop    = test_img_uncrop.reshape((h_uncrop, w_uncrop, n_bands))

    test_img_unnorm = test_img_uncrop[0:height, 0:width, 0:n_bands]

    if pan_flag:
        test_img_unnormn = np.zeros((height, width, 3), dtype='uint8')
        test_img_unnormn[:,:,0] = test_img_unnorm[:,:,0]
        test_img_unnormn[:,:,1] = test_img_unnorm[:,:,0]
        test_img_unnormn[:,:,2] = test_img_unnorm[:,:,0]
        test_img_unnorm=test_img_unnormn

    test_img_unnorm[128 <= distance_estimated * 255, 0] = 255
    test_img_unnorm[(127 < distance_estimated * 255) & (distance_estimated * 255 < 128), 2] = 255

    output_img = Image.fromarray(test_img_unnorm.astype(np.uint8))
    output_img.save(input_image_path + 'k.png')

    # Generating an additional image for conversion to shapefile
    test_img_unnorm[(127 < distance_estimated * 255) & (distance_estimated * 255 < 128), 0] = 127
    test_img_unnorm[distance_estimated * 255 <= 127, 0] = 0

    output_img_shp = Image.fromarray(test_img_unnorm.astype(np.uint8))

    inRaster = input_image_path + 'k_shp.png'

    output_img_shp.save(inRaster)

    reclassField = "VALUE"
    remap = RemapValue([[0, "NODATA"], [255, 1], ["NODATA", "NODATA"]])
    outReclassify = Reclassify(inRaster, reclassField, remap)
    if arcpy.Exists(os.getcwd()):
        arcpy.Delete_management(os.getcwd())

    outReclassify.save(temp_path)

    arcpy.RasterToPolygon_conversion(temp_path, output_path)

