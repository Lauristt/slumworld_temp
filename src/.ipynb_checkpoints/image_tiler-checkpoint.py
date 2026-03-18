"""Module contains the class for tiling satilite images and Class for reconstructing images.

The tiler class is designed to be used in conjunction with the dataloader. The tiler class breaks down
an image into a series of smaller tiles which can be used by the dataloader to feed a convolutional
neural network. 

To maximise the number of distinct inputs to the CNN the tiler class creates tiles that have twice the height 
and width  (4x area) of the input image for the CNN. This allows the dataloader class to sample cropped tiles from the 
padded tiles. For situations where sampling is not required (during validation and testing) the dataloader class will split
the padded tile into 4 smaller tiles of the correct input dimensions.

The class can split the tiled images into training, validation and test sets. It does this within the "info.csv" file it 
generates during tiling. The info.csv file is used when reconstructing images
"""

import os
import shutil
import numpy as np
import json
from numpy.lib.shape_base import tile
import pandas as pd

from PIL import Image
from skimage import io
import warnings
## some times we encounter the warning or error message saying the image size is too large
## we can use following command to deal with it
## https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None


class Tiler:
    """Tiles satilite images for use with pytorch CNN. Designed to be used with DataLoader class
        
        Takes inputs images (and optional label and mask) and converts them into a series of tiles for
        feeding into a CNN. Default locations for saved tiles is within a new directory within the image data 
        folder.

        Parameters
        -----------
        data_folder_path : str
            Full path to folder containing satilite input (and labels/mask if being used). Unless stated in the optional
            arguments the tiler will look for the filenames input_x.png, input_y.png, input_z.png (input, labels, mask)
        
        tile_size : int
            length of a sqaure tile. This should be the same tile size as the one being used for the CNN.
        
        training_split : list of floats
            List of length 3 defining how the data will be split into [Train, Validation, Test]. Example:
            [0.6, 0.2, 0.2]. If you do not want to use a test then let it's value equal 0 and ensure the training 
            and validation values sum to 1. Example: [0.8, 0.2, 0]
        
        use_labels : bool, default=True
            Boolean whether to use labels when tiling. If true the tiler will look for label image file and tile it
            along with the input image. As default it will look for input_y.png
        
        use_mask : bool, default=True
            Boolean on whether to use mask when tiling. The mask will remove areas of the input image (and therfore label)
            which shouldn't be tiled.

        Optional Parmeters
        -------------------
        x_filename : str, default=input_x.png
            Filename for input satilite image. Tiler will look for this name within the data folder
        
        y_filename : str, default=input_y.png
            Filename for label image.

        z_filename : str, default=input_z.png
            Filename for mask
        
        convert_slum_to_binary : bool, default=True
            Slum labels are currently using "signed distances". The tiler can converts these into 1 or 0 (slum or no slum) if 
            this is set to True.
        
        slum_threshold : int, default=64
            Slum signed distances is such that any point with a value above the slum_threshold is a slum. This value is used 
            to turned the signed distance labels into binary labels. Only need to chance if there has been a change is how label
            file is being calculated and how slums are defined.
        
        mask_threshold_percentage : float, default=0.2
            If the percentage of masked pixels within a tile is above this threshold than the tile is considered masked.

        padded_tile_size_multiple : int, default=2
            Padded tiles height and width will be this multiple greater than the tile size. 
            e.g. if tile_size=256 and padded_tile_multipe=2 then the padded tiles will be for size [512, 512]
        
        visualise_tile_labels : bool, default=False
            If True saves binary labels as 0 and 127 for not slum and slum respectively. This allows the labels to be visualised for 
            manually checking. If False labels are saved as 0 or 1. CNN expects inputs of 0 or 1 of label pixels.

        """

    def __init__(self, data_folder_path, tile_size, training_split, tile_labels=True, use_mask=True, **kwargs):
        self.data_folder_path = data_folder_path
        self.tile_size = tile_size
        self.training_split = training_split
        self.tile_labels = tile_labels
        self.use_mask = use_mask
        
        # Optional paramaters. Default filenames overriden if provided
        optional_params = {'x_filename': "input_x.png", 'y_filename':"input_y.png", "z_filename": "input_z.png",
                        "padded_tile_size_multiple": 2, "convert_slum_to_binary": True, "slum_threshold" : 64, "mask_threshold_percentage" : 0.2,
                        "visualise_tile_labels" : False}
        optional_params.update(kwargs)
        self.x_filename = optional_params["x_filename"]
        self.y_filename = optional_params["y_filename"]
        self.z_filename = optional_params["z_filename"]
        self.padded_tile_size_multiple = optional_params["padded_tile_size_multiple"]
        self.slum_threshold = optional_params["slum_threshold"]
        self.convert_slum_to_binary = optional_params["convert_slum_to_binary"]
        self.mask_threshold_percentage = optional_params["mask_threshold_percentage"]
        self.visualise_tile_labels = optional_params["visualise_tile_labels"]

        # calculate and set \the padded tile size
        self.padded_tile_size = int(self.tile_size * self.padded_tile_size_multiple)

        # set training/validation/split
        self.set_training_validation_test_split(training_split)

        # check if filenames exist
        files_in_data_folder = os.listdir(self.data_folder_path)

        if self.x_filename not in files_in_data_folder:
            raise FileNotFoundError(self.x_filename + " cannot be found in data folder")
        
        if self.y_filename not in files_in_data_folder and self.tile_labels:
            raise FileNotFoundError(self.y_filename + " cannot be found in data folder")
        
        if self.z_filename not in files_in_data_folder and self.use_mask:
            raise FileNotFoundError(self.y_filename + " cannot be found in data folder")
    
    def set_training_validation_test_split(self, training_split):
        """Converts the input list into variables needed to create training/validation/test split. Checks
        that the training_split sums to one and is of length 3. Sets necessary class attributes that will be used
        by other methods"""

        if not isinstance(training_split, list):
            raise TypeError("Training split must be a list of decimals defining the respective splits [training, validation, test]")

        if sum(training_split) != 1:
            raise ValueError("Training split must sum to 1")
        
        if len(training_split) != 3:
            raise ValueError("training_split must be a list of decimals numbers of length 3 [training, validation, test]")

        self.use_test_set = False
        
        self.training_set_splits = [training_split[0], 
                                    training_split[0] + training_split[1],
                                    training_split[0] + training_split[1] + training_split[2]]
        
        if self.training_split[2] != 0:
            self.use_test_set = True

    def load_satilite_images(self):
        """Loads input image as well as label and mask if they are in use
        
        Input x is forces into 3D array. It allows all methods to work on both RGB and 1-Channel images. If a 
        dummy channel has been added it is later removed before saving.
        """

        # atleast_3d turns [m x n] into [m x n x 1]
        self.input_x = np.atleast_3d(io.imread(self.data_folder_path + "/" +  self.x_filename ))

        if self.tile_labels:
            self.input_y = io.imread(self.data_folder_path + "/" +  self.y_filename )
        if self.use_mask:
            self.input_z = io.imread(self.data_folder_path + "/" +  self.z_filename )
    
    def make_images_same_size(self):
        """Sometimes input, labels and masks image size differ by a 1 or 2 pixels. This makes all images the same size cropping all images
        to the smallest height and width"""
        x_dim = self.input_x.shape
        y_dim = [np.inf, np.inf]
        z_dim = [np.inf, np.inf]

        if self.tile_labels:
            y_dim = self.input_y.shape
            
        if self.use_mask:
            z_dim = self.input_z.shape

        # find the smallest width and height for each image
        smallest_height = np.min([x_dim[0], y_dim[0], z_dim[0]])
        smallest_width = np.min([x_dim[1], y_dim[1], z_dim[1]])

        # resize the x input
        self.input_x = self.input_x[0:smallest_height, 0:smallest_width, :]

        # resize labels and mask if in use
        if self.tile_labels:
            self.input_y = self.input_y[0:smallest_height, 0:smallest_width]
        
        if self.use_mask:
            self.input_z = self.input_z[0:smallest_height, 0:smallest_width]
    
    def calculate_padding_dims(self):
        """Calculates the number of pixels image needs to be padded by in order for it to be divisable by the padded_tile_size 
        without remainder"""

        height_padding = self.input_x.shape[0] % self.padded_tile_size
        width_padding = self.input_x.shape[1] % self.padded_tile_size

        return [height_padding, width_padding]

    
    def pad_images(self):
        """Adds vertical and horizonal row to the right and bottom of an image in order for it be divisable by padded_tile_size without
        remainder
        """

        self.padding_dims = self.calculate_padding_dims()

        self.input_x = np.pad(self.input_x, ((0, self.padding_dims[0]), (0, self.padding_dims[1]), (0, 0)), 'reflect')

        if self.tile_labels:
            self.input_y = np.pad(self.input_y, ((0, self.padding_dims[0]), (0, self.padding_dims[1])),'reflect')
        
        if self.use_mask:
            self.input_z = np.pad(self.input_z, ((0, self.padding_dims[0]), (0, self.padding_dims[1])),'reflect')
        

    def create_tile_coordinates(self):
        """ Creates the top left tile coordinate. Since each tile is a sqaure we only need the top left + tile size describe 
        location of every tile. 
        """
        
        # Create array marking each step along y-axis and x-axis
        y = np.arange(0, self.input_x.shape[0] - self.tile_size + 1, step=self.padded_tile_size)
        x = np.arange(0, self.input_x.shape[1] - self.tile_size + 1, step=self.padded_tile_size)
        
        # Combine both arrays into a grid to get coordinates of every top left corner a tile.
        xx, yy = np.meshgrid(x, y)
        
        # Convert grid into coordinate pairs. Note the height (y-axis) is the first value and width (x-axis) is the 2nd value.
        self.coordinates = np.array((yy.ravel(), xx.ravel())).T
    
    
    def lookup_which_dataset(self, randnum):
        """Sequentially checks if the randnum is in Train, Validation, or Test set. 
        
        A train/val/test split of [0.6, 0.2, 0.2] has be converted into [0.6, 0.8, 1] on class init. If randnum is below 0.6 return Train, 
        if below 0.8 return Validation, if below 1 return Test"""
        training_sets = ["Train", "Validation", "Test"]

        for i in range(3):
            if randnum < self.training_set_splits[i]:
                return training_sets[i]


    def create_saved_tile_location(self):

        # Create new directory with tilesize and overlap percentage in its name. Needed for reconstruction.
        tiler_dir_name = "TiledImage_Paddedtilesize_" + str(self.padded_tile_size) + "Tilesize_" + str(self.tile_size)
        self.save_path = os.path.join(self.data_folder_path, tiler_dir_name)

        # delete folder if it exists
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

        self.tile_folders = ["input_x_tiled/"]
        os.makedirs(os.path.join(self.save_path, self.tile_folders[0]))
        
        if self.tile_labels:
            self.tile_folders.append("input_y_tiled/")
            os.makedirs(os.path.join(self.save_path, self.tile_folders[1]))
        
        # create folder for masked tiles
        if self.use_mask:
            self.mask_folder = "masked_tiles/"
            os.makedirs(os.path.join(self.save_path, self.mask_folder))


    def is_tile_masked(self, current_coord):
        # checks the mask to see if the current tile is under the mask
        mask_tile = self.input_z[current_coord[0]:current_coord[0] + self.padded_tile_size, current_coord[1]:current_coord[1]+self.padded_tile_size]

        non_zero_values = np.count_nonzero(mask_tile)
        percentage_masked = 1 - (non_zero_values / mask_tile.shape[0] **2)

        if percentage_masked > self.mask_threshold_percentage:
            return True
        else:
            return False

    @staticmethod
    def return_image_to_correct_dimensions(array):
        """Dummy dimensions are added to 2D images. Need to remove them before saving"""

        if len(array.shape) == 3 and array.shape[-1] == 1:
            return array[:, :, 0]
        else:
            return array

    def convert_y_labels_to_binary(self):
        """ Turns the label binary by replace any value that is greater than the self.slum_threshold with a slum_pixel_value. Any value less than it will be set to 0.
        """

        slum_pixel_value = 1
        if self.visualise_tile_labels:
            slum_pixel_value = 127
            
        if self.convert_slum_to_binary:
            self.input_y = np.where(self.input_y > self.slum_threshold, slum_pixel_value, 0).astype("uint8")

            # slum threshold has now changed so need to update
            self.slum_threshold = slum_pixel_value

    def calculate_slum_coverage(self, array):
        # TODO fix this for when slum are not used
        if self.slum_threshold > 1:
            array = np.where(array >= self.slum_threshold, 1, 0)
        
        slum_percentage = array.sum() / (self.padded_tile_size **2)

        return slum_percentage
    
    def create_split_coordinates(self, tile_coordinates):
        """splits a padded tile of size padded_tile_size into smaller tiles of size tile_size"""
        
        # Create array marking each step along y-axis and x-axis
        y = np.arange(tile_coordinates[0], tile_coordinates[0] + self.padded_tile_size, step=self.tile_size)
        x = np.arange(tile_coordinates[1], tile_coordinates[1] + self.padded_tile_size, step=self.tile_size)
        
        # Combine both arrays into a grid to get coordinates of every top left corner a tile.
        xx, yy = np.meshgrid(x, y)
        
        # Convert grid into coordinate pairs. Note the height (y-axis) is the first value and width (x-axis) is the 2nd value.
        return np.array((yy.ravel(), xx.ravel())).T
    
    def calculate_normalising_values(self, image):
        """Calculates the normalising constant for each channel within the image"""

        img_mean = []
        img_std = []
        for i in range(image.shape[-1]):
            # image has only one channel
            img_mean.append(image[:, :, 0].mean())
            img_std.append(image[:, :, 0].std())
        
        return img_mean, img_std

    def tile(self):
        """Function uses initialised variables to tile satilite images"""
        
        print("Loading images for tiling...")
        self.load_satilite_images()
        print("Tiling started...")

        img_mean, img_std = self.calculate_normalising_values(self.input_x)

        self.make_images_same_size()
 
        # Pad the right and lower border of inputs to make them divisable by padded_tile_size without remainder
        self.pad_images()

        # Change slum_pixel_value to create binary tiles with higher contrast (e.g. 127). The CNN is expecting labels to have 0 or 1 values.
        self.convert_y_labels_to_binary()

        # calculate the top left-hand coordinate of each tile.
        self.create_tile_coordinates()

        # create all the folders needed to save the tiles into training/validation/test
        self.create_saved_tile_location()

        # pandas dataframe captures save location and information on all tiles. Preallocate maximum size and trim at the end.
        df = pd.DataFrame(index=range(len(self.coordinates) * self.padded_tile_size_multiple),
        columns=['x_location', 'y_location', 'training_set', "slum_coverage"])

        tile_number = 0
        # tile image and save
        for tile_coor in self.coordinates:
            
            # get a padded x tile
            current_x_tile = self.input_x[tile_coor[0]:tile_coor[0] + self.padded_tile_size, tile_coor[1]:tile_coor[1]+self.padded_tile_size, :]
            location_str = str(tile_coor[0]) + "_" + str(tile_coor[1])
            im_x = self.return_image_to_correct_dimensions(current_x_tile)

            if self.use_mask and self.is_tile_masked(tile_coor):
                # if tile is masked then save to mask folder

                file_name = self.mask_folder + location_str + ".png"
                io.imsave(os.path.join(self.save_path, file_name), im_x, check_contrast=False)
                df.iloc[tile_number] = [file_name, "", "Mask", ""]

                tile_number +=1
            
            else:
                # create a random number from 0 to 1 to decide which set it will go to
                randint = np.random.uniform()
                train_set = self.lookup_which_dataset(randint)

                if train_set == "Train":
                    
                    # process the labels first. slum identifier is needed before saving x tile.
                    if self.tile_labels:
                        current_y_tile = self.input_y[tile_coor[0]:tile_coor[0] + self.padded_tile_size, tile_coor[1]:tile_coor[1]+self.padded_tile_size]

                        slum_coverage = self.calculate_slum_coverage(current_y_tile)

                        if slum_coverage > 0:
                            # Add identifier to the saved name foeasy mannual checking
                            location_str = location_str + "_slum"
                        
                        # save label tile
                        label_save_path = self.tile_folders[1] + location_str + ".png"
                        io.imsave(os.path.join(self.save_path, label_save_path), current_y_tile, check_contrast=False)
                        
                        # Update info.csv file
                        df.at[tile_number, "y_location"] = label_save_path
                        df.at[tile_number, "slum_coverage"] = slum_coverage
                    
                    # save x tile to training folder
                    x_save_path = self.tile_folders[0] + location_str + ".png"
                    io.imsave(os.path.join(self.save_path, x_save_path), im_x)

                    # update the info.csv with tile location
                    df.at[tile_number, "x_location"] = x_save_path
                    df.at[tile_number, "training_set"] = train_set

                    tile_number +=1
                
                else:
                    # This tile will enter either the validation or testing set and therefore should not be padded. Split the tile into equal
                    # subtiles of tile tile_size and save each one separately.
                    split_coor = self.create_split_coordinates(tile_coor)

                    for sub_coor in split_coor:
                        sub_x_tile = self.input_x[sub_coor[0]:sub_coor[0] + self.tile_size, sub_coor[1]:sub_coor[1]+self.tile_size, :]
                        location_str = str(sub_coor[0]) + "_" + str(sub_coor[1])

                        # process the labels first. slum identifier is needed before saving x tile.
                        if self.tile_labels:
                            sub_y_tile = self.input_y[sub_coor[0]:sub_coor[0] + self.tile_size, sub_coor[1]:sub_coor[1]+self.tile_size]

                            slum_coverage = self.calculate_slum_coverage(sub_y_tile)

                            if slum_coverage > 0:
                                # Add identifier to the saved name foeasy mannual checking
                                location_str = location_str + "_slum"
                            
                            # save label tile
                            label_save_path = self.tile_folders[1] + location_str + ".png"
                            io.imsave(os.path.join(self.save_path, label_save_path), sub_y_tile, check_contrast=False)
                            
                            # Update info.csv file
                            df.at[tile_number, "y_location"] = label_save_path
                            df.at[tile_number, "slum_coverage"] = slum_coverage
                        
                        # save x tile to training folder
                        im_x = self.return_image_to_correct_dimensions(sub_x_tile)
                        x_save_path = self.tile_folders[0] + location_str + ".png"
                        io.imsave(os.path.join(self.save_path, x_save_path), im_x)

                        # update the info.csv with tile location
                        df.at[tile_number, "x_location"] = x_save_path
                        df.at[tile_number, "training_set"] = train_set
                        
                        tile_number +=1

        # drop the empty preallocated rows and save
        df.dropna(subset = ["x_location"], inplace=True)
        df.to_csv(self.save_path + "/info.csv")

        # save the normalising constant along with summary information
        summary = {
            "mean" : img_mean,
            "std" : img_std,
            "num_of_channels" : int(self.input_x.shape[-1]),
            "padded_image_size" : self.input_x.shape,
            "padding_dimensions" : self.padding_dims,
            "padded_tile_size" : self.padded_tile_size,
            "tile_size" : self.tile_size,
            "num_of_tiles" : tile_number,
            "training_set_size" : len(df[df["training_set"] == "Train"]),
            "validation_set_size" : len(df[df["training_set"] == "Validation"]),
            "test_set_size" : len(df[df["training_set"] == "Test"])      
        }

        json.dump(summary, open(self.save_path + "/summary.json", 'w'))
        

     
        print("Tiling complete. Tiles saved in " + self.save_path)


class Reconstruct_image():
    pass


if __name__ == "__main__":

    # "/home/raza/code/slumworldML/data/round_73_Briana_PAN_inputs/MD_python"
    tiler = Tiler("/home/raza/code/slumworldML/data/round_75_Briana_MUL_inputs/MD",
                    tile_size=256,
                    training_split=[0.8, 0.2, 0])
    
    # split = tiler.create_split_coordinates([1000, 2000])
    # print(split)
    tiler.tile()








