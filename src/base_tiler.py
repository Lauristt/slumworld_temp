"""Module contains class for tiling Images. ImageTiler breaks down a large image into a series of smaller tiles
defined by tile_size. If the image is not exactly divisable by the tile_size, the original image is padded 
(reflection padding) and then tiled. When using the tile_and_save function folder in which the tiles will be saved
must already be created. Imagetiler class includes a function to reconstruct an image. The reconstuct method does
not require the ImageTiler class to be initialised.

    Usage:

    Tiling an image and saving it to a folder. The folder in which the image will be saved into must be created
    prior to tiling.

    >>> tiler = ImageTiler(image_path="path/to/image.png", tile_size=512)
    >>> tiler.tile_and_save(save_location="path/to/save/location")

    Reconstructing an image
    If the image was padded during tiling then original size is lost. Use target size to trim the reconstructed
    image to a desired size. The original image size has to obtained separately.

    >>> ImageTiler.reconstruct_image(tile_folder_path="path/to/tiled_image", 
                                        output_filename="path/and/filename/reconstructed.png", 
                                        target_size=[15683, 2908])


"""

import sys
import os
# sys.path.append("..")
# __package__ = os.path.dirname(sys.path[0])
import shutil
import numpy as np
import json
import pandas as pd
from PIL import Image
from skimage import io
import warnings
import pdb
## some times we encounter the warning or error message saying the image size is too large
## we can use following command to deal with it
## https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None



class ImageTiler():
    """Tiles an image into a series of smaller tiles defined by tile size. 
    If image height and/or with is not multiple of tile_size then the image is padded with a reflection."""

    def __init__(self, image_path, tile_size, is_label_image=False):
        self.tile_size = tile_size
        
        self.image = io.imread(image_path)

        self.img_size = [self.image.shape[0], self.image.shape[1]]
        self.three_d = True if len(self.image.shape) == 3 else False
        self.is_label = is_label_image
        self.dtype = self.image.dtype
        
        # Pad the image so it is a multiple of tile_size
        self.padding_dims = self.calculate_padding_dims(self.tile_size, self.image.shape)
        if self.three_d:
            self.image = np.pad(self.image, ((0, self.padding_dims[0]), (0, self.padding_dims[1]), (0, 0)), 'reflect')
        else:
            self.image = np.pad(self.image, ((0, self.padding_dims[0]), (0, self.padding_dims[1])), 'reflect')

    @staticmethod
    def calculate_padding_dims(tile_size, image_shape):
        """Calculates the number of pixels image needs to be padded by in order for it to be divisable by the padded_tile_size 
        without remainder"""

        height_padding = (tile_size - image_shape[0] % tile_size) * (image_shape[0] % tile_size > 0)
        width_padding = (tile_size - image_shape[1] % tile_size) * (image_shape[1] % tile_size > 0)

        return [height_padding, width_padding]
        
    def coordinate_generator(self):
       
        num_of_tiles_high = self.image.shape[0]//self.tile_size
        num_of_tiles_wide = self.image.shape[1]//self.tile_size

        for i in range(num_of_tiles_high):
            for j in range(num_of_tiles_wide):
                y1 = i * self.tile_size
                y2 = y1 + self.tile_size

                x1 = j * self.tile_size
                x2 = x1 + self.tile_size

                yield y1, y2, x1, x2

    def tile_generator(self):

        coordinates = self.coordinate_generator()

        for y1, y2, x1, x2 in coordinates:
            if not self.is_label:
                if self.three_d:
                    yield self.image[y1:y2, x1:x2, :], [y1, x1]
                else:
                    yield np.tile(self.image[y1:y2, x1:x2][:,:,None], [1,1,3]), [y1, x1]
            else:
                yield self.image[y1:y2, x1:x2], [y1, x1]
                
    def tile_and_save(self, save_location):
        """Save location must created prior to calling function"""

        tile_gen = self.tile_generator()
        
        for tile, location in tile_gen:
            save_str = os.path.join(save_location, self.create_save_str(location))
            io.imsave(save_str, tile, check_contrast=False)

    @staticmethod
    def create_save_directory(save_location, overwrite=False):
        """WARNING: overwrite will delete the directory and create a new one."""
        try:
            os.makedirs(save_location)
        except FileExistsError:
            if overwrite:
                shutil.rmtree(save_location)
                os.makedirs(save_location)
            else:
                print("Warning! Output directory exists already. Results may overide existing files.")

    @staticmethod
    def create_save_str(location, file_extension=".png"):
        return str(location[0]) + "_" + str(location[1]) + file_extension
    
    @staticmethod
    def reconstruct_image(tile_folder_path, output_filename, target_size=None, colourize=False):
        """Reconstruct full map from individual prediction tiles.
        Call as static method on tiled folder to reconstruct image
        Args:
            tile_folder_path:           path, folder containing prediction tiles 
            output_filename:            path, desired filename for the reconstructed image
            target_size:                tuple(int,int), size of the original image (and of the reconstructed)
                                        required in order to remove any padding added during tiling
                                        If none the reconstructed image will include the padding
                                        [default: None] 
            colourize:                  boolean, if True the reconstructed map will multiplied by 255 to be visible in image editors
                                        [default: False]
            """

        tile_locations = [os.path.join(tile_folder_path, tile_name) for tile_name in os.listdir(tile_folder_path) \
                                                        if (tile_name.endswith('.png') or tile_name.endswith('jpg'))]
        assert (os.path.exists(tile_folder_path)) and (len(tile_locations)>0), print("Error! Supplied tile_folder_path does not exist or does not contain any image files. Aborting...")

        # load a single image to get tile info
        a_tile = io.imread(tile_locations[0])
        tile_size = a_tile.shape[0]
        three_d = True if len(a_tile.shape) == 3 else False
        dtype = a_tile.dtype

        # Prealocate
        height, width = ImageTiler.find_image_size_from_tiles(tile_folder_path, tile_size)
        if three_d:
            img = np.zeros([height, width, 3], dtype=dtype)
        else:
            img = np.zeros([height, width], dtype=dtype)
        for filename in tile_locations:

            # Get the positions and update
            y, x = ImageTiler.extract_coordinate(filename)
            tile = io.imread(filename)

            # # =================== DEBUGGING ===================
            # print(f"DEBUG: Processing {filename}")
            # print(f"DEBUG: y={y} (type: {type(y)})")
            # print(f"DEBUG: x={x} (type: {type(x)})")
            # print(f"DEBUG: tile_size={tile_size} (type: {type(tile_size)})")
            # # =============================================================

            if three_d:
                img[y:y+tile_size, x:x+tile_size, :] = tile
            else:
                img[y:y+tile_size, x:x+tile_size] = tile

        if target_size is not None:
            if three_d:
                img = img[0:target_size[0], 0:target_size[1], :]
            else:
                img = img[0:target_size[0], 0:target_size[1]]
        if colourize:
            img *= 255
        ImageTiler.create_save_directory(os.path.dirname(output_filename), overwrite=False)
        
        io.imsave(output_filename, img, check_contrast=False)

    @staticmethod
    def extract_coordinate(coordinate_str):

            split_str = coordinate_str.split(os.sep)[-1]
            coor = os.path.splitext(split_str)[0]
            coor = coor.split("_")
            return int(coor[0]), int(coor[1])
    
    @staticmethod
    def find_image_size_from_tiles(tile_folder_path, tile_size):
        tile_filenames = [tilename for tilename in os.listdir(tile_folder_path) if (tilename.endswith('png') or tilename.endswith('jpg'))]

        # Find how big the tiled image was by extracting the largest y and x value.
        y_max = 0
        x_max = 0
        for tile_name in tile_filenames:
            y, x = ImageTiler.extract_coordinate(tile_name)
            y_max = y if y_max < y else y_max
            x_max = x if x_max < x else x_max
        
        y_max += tile_size
        x_max += tile_size

        return y_max, x_max
