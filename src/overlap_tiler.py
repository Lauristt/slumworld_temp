import sys
import os
import shutil
import json
import numpy as np
from skimage import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pdb

class OverlapTiler():
    """OverlapTiler class takes an image, pads it and splits it into tiles. The tiles
    are produced with 50% overlap. The class provides a method for reconstructing
    the original image from the tiles. This is done by extracting only the "middle"
    part of each of the generated tiles (50% of the height and width of the tile size)
    In that way, during inference the model can have more context for the predicted part
    of the tile, without changing the shape of the original image.
    Args:
        1. Tiling:
            input_image_path:               str, path to the location of the image to be tiled
            output_folder:                  str, path to the location of the folder where the tiles will be stored
            tile_size:                      int, the dimension of the tiles (tiles have square shape)
            info_json_folder:               str, path to the location of the folder where json summary file will be stored
                                            if None it will be placed inside the output_folder [default:None]
            is_label_image:                 boolean, indicates whether the image to be tiled is a label (2D) or not  [default:False]
        2. Reconstruction:
            tiling_info_json:               str, path to saved info json (saved during tiling)
                                            if reconstructing model predictions tiles, one should again use the json produced during tiling
            tile_folder:                    str, path to the folder with the tiles that will be joined together for reconstruction/,
                                            if None, then the output location indicate in the tiling_info_json will be used [default: None]
            output_path:                    str, path to the output folder where the reconstructed map will be stored
            output_filename:                str, name for the reconstructed image [default: 'reconstructed_image.png']
            make_visible:                   boolean, if True, then image pixels will be multiplied with 255 for being visible in standard
                                            imaging apps [default:True]
    Usage:
        # Tiling
        >>>  overlapTiler = OverlapTiler(input_image_path="path/to/image.png",output_folder="path/to/tile_folder",
                                         info_json_folder="path/to/info_folder",tile_size=512)
        >>> overlapTiler.tile_and_save()
        # Reconstruction
        >>> overlapTiler.reconstruct_image(tiling_info_json="path/to/info_folder/image_info_path",tile_folder="path/to/tiles_for_reconstruction/",
                                           output_path="path/to/output_folder", output_filename='reconstructed_image.png', make_visible=True)
    """

    def __init__(self, input_image_path, output_folder, tile_size, 
                 info_json_folder=None, is_label_image=False):

        self.tile_size = tile_size
        self.extra_pad_size = int(0.25 * self.tile_size)

        self.image = io.imread(input_image_path)
        self.input_image_path = input_image_path
        self.original_image_size = [self.image.shape[0], self.image.shape[1]]
        self.image3d = True if len(self.image.shape) == 3 else False

        self.is_label = is_label_image
        self.dtype = self.image.dtype

        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                print("Created output folder.")
            except Exception as Error:
                print("Error! Could not create output folder. Error log:", Error)
                print("Exiting...")
        else:
            existing_files = os.listdir(output_folder)
            if any([f.endswith('.png') or f.endswith('.jpg') for f in existing_files]):
                print("Output folder exists and contains image files.")
                print("Please provide a folder without any image files in it.")
                print("Aborting operation to avoid accidental loss of data...")
                sys.exit(1)
        if info_json_folder:
            self.info_json_folder = info_json_folder
            if not os.path.exists(info_json_folder):
                try:
                    print("Could not find info_json_folder . Creating one")
                    os.makedirs(info_json_folder, exist_ok=True)
                    print("Created info_json_folder.")
                except Exception as Error:
                    print("Error! Could not create info_json_folder. Error log:", Error)
                    print("Exiting...")
        else:
            self.info_json_folder = output_folder
        self.padded_image_path = os.path.join(self.output_folder, 'padded_image.jpg')
        self.prepare_image()

    def prepare_image(self):
        """Makes image divisible by tile size by padding at the right and bottom of it with reflection.
        Adds extra 1/4 of tile size reflection padding on all sides."""

        # Calculate height and width padding in order for the image to be divisible by the tile size.
        height_padding = (self.tile_size - self.original_image_size[0] % self.tile_size) * (self.original_image_size[0] % self.tile_size > 0)
        width_padding = (self.tile_size - self.original_image_size[1] % self.tile_size) * (self.original_image_size[1] % self.tile_size > 0)

        self.padded_image_size = [self.original_image_size[0] + height_padding, self.original_image_size[1] + width_padding]
        self.quarter_padded_image_size = [self.padded_image_size[0] + (2 * self.extra_pad_size), self.padded_image_size[1] + (2 * self.extra_pad_size)]

        if self.image3d:
            self.image = np.pad(self.image, ((self.extra_pad_size, (height_padding+self.extra_pad_size)),(self.extra_pad_size, (width_padding+self.extra_pad_size)),(0,0)), 'reflect')
        else:   
            self.image = np.pad(self.image, ((self.extra_pad_size, (height_padding+self.extra_pad_size)), (self.extra_pad_size, (width_padding+self.extra_pad_size))), 'reflect')

        io.imsave(self.padded_image_path, self.image, check_contrast=False)

    def coordinate_generator(self):

        height_tiles = (self.image.shape[0]//self.tile_size) * 2
        width_tiles = (self.image.shape[1]//self.tile_size) * 2

        self.total_height_tiles = height_tiles
        self.total_width_tiles = width_tiles

        for h_idx in range(height_tiles):
            for w_idx in range(width_tiles):
                h1 = int((h_idx/2) * self.tile_size)
                h2 = h1 + self.tile_size
                
                w1 = int((w_idx/2) * self.tile_size)
                w2 = w1 + self.tile_size

                yield h1, h2, w1, w2

    def tile_generator(self):

        tile_coordinates = self.coordinate_generator()

        for h1, h2, w1, w2 in tile_coordinates:
            if not self.is_label:
                if self.image3d:
                    yield self.image[h1:h2, w1:w2, :], [h1, w1]
                else:
                    yield np.tile(self.image[h1:h2, w1:w2][:,:,None], [1,1,3]), [h1, w1]
            else:
                yield self.image[h1:h2, w1:w2], [h1, w1]
    
    def create_save_str(self, coordinates, file_extension=".png"):
        return str(coordinates[0]) + "_" + str(coordinates[1]) + file_extension

    def tile_and_save(self):
        tile_generator = self.tile_generator()
        for tile, coordinates in tile_generator:
            save_str = os.path.join(self.output_folder, self.create_save_str(coordinates))
            io.imsave(save_str, tile, check_contrast=False)
        
        self.save_image_info()

        self.clean_up()

    def clean_up(self):
        os.remove(self.padded_image_path)
 
    def save_image_info(self):
        image_summary = {
            "input_image_path" : os.path.abspath(self.input_image_path),
            "is_image_3d" : self.image3d,
            "tile_size" : int(self.tile_size),
            "original_input_size" : [int(self.original_image_size[0]), int(self.original_image_size[1])],
            "padded_image_size" : [int(self.padded_image_size[0]),int(self.padded_image_size[1])],
            "output_tiles_folder" : os.path.abspath(self.output_folder)
        }

        with open(self.info_json_folder+"/overlap_tiling_info.json", "w") as info_file:
            json.dump(image_summary, info_file, indent=4)
            info_file.close()

    @staticmethod
    def extract_coordinates(coordinate_str):
    
        split_str = coordinate_str.split(os.path.sep)[-1]
        coor = os.path.splitext(split_str)[0]
        coor = coor.split("_")
        return int(coor[0]), int(coor[1])

    @staticmethod
    def reconstruct_image(tiling_info_json, tile_folder=None, output_path=None, 
                          output_filename='reconstructed_image.png', make_visible=True):
        """Reconstructs image from tiles."""
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            
        if output_path is None:
            output_path = os.path.join(os.path.dirname(), output_filename)
        else:
            output_path = os.path.join(output_path, output_filename)

        with open(tiling_info_json) as jsonFile:
            jsonObject = json.load(jsonFile)

        if tile_folder is None:
            tile_folder = jsonObject['output_tiles_folder']
        original_image_size = [jsonObject['original_input_size'][0], jsonObject['original_input_size'][1]]
        padded_image_size = [jsonObject['padded_image_size'][0], jsonObject['padded_image_size'][1]]
        tile_size = jsonObject['tile_size']
        extra_pad_size = int(0.25 * tile_size)
        
        tile_locations = [os.path.join(tile_folder, tile_name) for tile_name in os.listdir(tile_folder) 
                            if (tile_name.endswith('.png') or tile_name.endswith('jpg'))]

        # load a single image to get tile info
        tile = io.imread(tile_locations[0])
        dtype = tile.dtype
        image3d = len(tile.shape) == 3

        if image3d:
            image = np.zeros([padded_image_size[0], padded_image_size[1], 3], dtype=dtype)
        else:
            image = np.zeros([padded_image_size[0], padded_image_size[1]], dtype=dtype)

        patch_size = int(tile_size/2)

        for filename in tile_locations:

            h, w = OverlapTiler.extract_coordinates(filename)

            image_tile = io.imread(filename)

            if image3d:
                image[h:h+patch_size, w:w+patch_size, :] = image_tile[extra_pad_size:extra_pad_size+patch_size, extra_pad_size:extra_pad_size+patch_size, :]
            else:
                image[h:h+patch_size, w:w+patch_size] = image_tile[extra_pad_size:extra_pad_size+patch_size, extra_pad_size:extra_pad_size+patch_size]

        if image3d:
            final_image = image[0:original_image_size[0], 0:original_image_size[1], :]
        else:
            final_image = image[0:original_image_size[0], 0:original_image_size[1]]
        if make_visible:
            final_image *= 255
        io.imsave(output_path, final_image, check_contrast=False)