"""Module contains the class for tiling satellite images and Class for reconstructing images.
The tiler class is designed to be used in conjunction with the dataloader. The tiler class breaks down
an image into a series of smaller tiles which can be used by the dataloader to feed a convolutional
neural network. 
To maximise the number of distinct inputs to the CNN the tiler class creates tiles that have twice the height 
and width  (4x area) of the input image for the CNN. This allows the dataloader class to sample cropped tiles from the 
padded tiles. For situations where sampling is not required (during validation and testing) the dataloader class will split
the padded tile into 4 smaller tiles of the correct input dimensions.
The class can split the tiled images into training, validation and test sets. It does this within the "info.csv" file it 
generates during tiling. The info.csv file is used when reconstructing images
    Usage:
    >>> from transforms_loader import create_transform, TRAINING_TRANSFORMS
    >>> dpath = '/home/minas/slumworld/data/input_raw/MD_MUL_75_Briana'
    >>> opath = '/home/minas/slumworld/data/tiled/MD_MUL_75_Briana_New'
    >>> tiler = CNNTiler(tile_size=512, save_path=opath)
    # tile inputs
    >>> tiler.tile_inputs(x_input_path=dpath+'/input_x.png', y_input_path=dpath+'/input_y.png', labels2binary=True)
    # Use the tiled inputs to create a standard (train-val-test) dataset
    >>> tiler.create_standard_dataset(training_split=[0.7,0.15,0.15], mask=os.path.join(dpath, "input_z.png"), labels2binary=True)
    # Use the tiled inputs to create a k-fold dataset
    >>> tiler.create_kfold_dataset(num_of_folds=5, 
                            test_set_frac=0.2,
                            mask=os.path.join(dpath, "input_z.png"))
    # calculate balancing statistics for training
    >>> transformation = create_transform(TRAINING_TRANSFORMS, mean=[1,1,1], std=[1,1,1]) # Use dummy mean and std for now
    >>> tiler.calculate_tile_statistics(transformation, num_of_samples=200)
    # Use the tiled inputs and an existing standard dataset.csv (with calculated statistics) to create a k-fold dataset
    >>> tiler.create_kfold_dataset(num_of_folds=5, 
                            test_set_frac=0.2,
                            mask=os.path.join(dpath, "input_z.png"),
                            existing_df=pd.read_csv(os.path.join(opath,'dataset.csv')) )    
    # TO RECONSTRUCT AN IMAGE FROM TILES
    >>> ImageTiler.reconstruct_image(path_to_predicted_tiles, full path of file to save to )
"""

import sys
import os
from pathlib import Path
# sys.path.append("..")
# __package__ = os.path.dirname(sys.path[0])
import gc
# import shutil
import numpy as np
import json
import pdb
import torch
from numpy.lib.shape_base import tile
from numpy.random.mtrand import triangular
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import io
from torchvision.io import read_image
# import warnings
try:
    from slumworldML.src.base_tiler import ImageTiler
    from slumworldML.src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
    from slumworldML.src.custom_transformations import BinarizeLabels
except Exception as Err1:
    try:
        from src.base_tiler import ImageTiler
        from src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
        from src.custom_transformations import BinarizeLabels
    except Exception as Err2:
        try:
            from base_tiler import ImageTiler
            from transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
            from custom_transformations import BinarizeLabels
        except Exception as Err3:
            try:
                from .src.base_tiler import ImageTiler
                from .src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
                from .src.custom_transformations import BinarizeLabels
            except Exception as Err3:
                from ..src.base_tiler import ImageTiler
                from ..src.transforms_loader import create_transform, TRAINING_TRANSFORMS_BASIC
                from ..src.custom_transformations import BinarizeLabels

## some times we encounter the warning or error message saying the image size is too large
## we can use following command to deal with it
## https://stackoverflow.com/questions/25705773/image-cropping-tool-python
Image.MAX_IMAGE_PIXELS = None


class CNNTiler():
    """Tiles input, labels and prepares data for training."""
    def __init__(self, tile_size, save_path=None, mask_threshold=0.99):
        '''
            Args: 
                tile_size:      int, the tile size to use (the Neural Network will actually be fed with a tile
                                of half this tile_size, due to the augmentation method employed here)
                save_path:      str, the path to the directory where the tiles will be saved. The directory will be created if non-existent.
                                Two sub-directories will also be created, "tiled_input" and "tiled_labels" to hold the corresponding
                                image and label (if the latter exist) tiles.
                mask_threshold: float, the minimum % of masked pixels in a tile in order to mask it out of the calculations
        '''
        self.save_path = save_path
        self.tile_size = tile_size
        self.save_folders = ["tiled_input", "tiled_labels"]
        self.mask_threshold_percentage = mask_threshold
        self.slum_threshold = 64
        self.csv_save_name = "dataset.csv"
        self.input_size = -1
        
        # create the folder if it does not exist
        if save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def tile_inputs(self, x_input_path, y_input_path, labels2binary=True):
        
        # Tile input and labels
        img_sizes = []
        self.labels2binary = labels2binary
        if y_input_path is not None:
            is_label = 0
            for img_path, save_folder in zip([x_input_path, y_input_path], self.save_folders):
                save_dir = os.path.join(self.save_path, save_folder)
                # check if we are tilling labels (output tiles should be 2-D, no colour channels)
                tiler = ImageTiler(img_path, self.tile_size, is_label_image=is_label==1)
                tiler.create_save_directory(save_dir, overwrite=True)
                tiler.tile_and_save(save_dir)
                img_sizes.append(tiler.img_size)

                del tiler
                gc.collect()

                is_label += 1

            if img_sizes[0] != img_sizes[1]:
                raise ValueError("Inputs and labels have different heights and/or widths. Trim images first. [input, labels] sizes are:", img_sizes)
            
            self.input_size = img_sizes[0]

            # Turn labels to binary
            if labels2binary:
                print("Binarizing Labels ...")
                label_path = os.path.join(self.save_path, self.save_folders[1])
                label_filenames = os.listdir(label_path)
                for filename in label_filenames:
                    a_label = io.imread(os.path.join(label_path, filename))
                    a_label = CNNTiler.convert_y_labels_to_binary(a_label)
                    io.imsave(os.path.join(label_path, filename), a_label, check_contrast=False)
        else:
    
            save_dir = os.path.join(self.save_path, self.save_folders[0])
            tiler = ImageTiler(x_input_path, self.tile_size, is_label_image=False)
            tiler.create_save_directory(save_dir, overwrite=True)
            tiler.tile_and_save(save_dir)
            img_sizes = [tiler.img_size]

            del tiler
            gc.collect()

            self.input_size = img_sizes[0]

    @staticmethod
    def _image_files_list(folder):
        assert os.path.exists(folder), f"Error! Could not locate supplied folder ({folder})."
        assert len(os.listdir(folder)) > 0, f"Error! Supplied folder ({folder}) does not contain any files."
        return [ f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]


    def dataset_from_folder(self, img_folder, label_folder=None, training_split=[0.6,0.2,0.2], img2label_function=None,
                                save_path=None, dataset_name='dataset.csv', calculate_stats=False, TRANSFORMS=None):
        '''Convinence function that handles creation of a dataset from a folder of images tiles and, optionally a folder of label tiles.j
            Args:
                img_folder          str, path to the folder holding the satellite image tiles 
                label_folder        str or None, path to the folder holding the label image tiles[default: None]
                training_split      list[float, float, float], train-validation-test split percentage [default: [0.6,0.2,0.2]]
                img2label_function  callable or None, a function that maps the names of the image tiles to the name of the labels tiles,
                                    can be None if the image and labels have the same names [default: None]. 
                                    if supplied the function should consume an input_image_name and yield the correspsongin label_image_name,
                                    e.g. label_tile_name = img2label_function(img_tile_name)
                save_path           str or None, location of the folder in which the dataset.csv file will be saved [default: None]. 
                dataset_name        str, the name of the dataset.csv file that will be created [default: 'dataset.csv']
                calculate_stats     boolean, if set to True the function will ran the balaning statistics calculation 
                                    This is REQUIRED FOR TRAINING but not for evaluating [default: False] 
                TRANSFORMS          pytorch transforms, the kind of joint transports that will be used for training (required for calculate_stats)
                                    if None, a set of basic transforms will be used [default: None]
            Returns:
                nothing, saves a dataset.csv file 
            Usage:
                >>> tiler = CNN_tiler(tile_size=512)
                >>> tiler.dataset_from_folder(img_folder=img_folder, label_folder=label_folder, training_split=[0.6,0.2,0.2], 
                                                  save_path=/path/to/save, dataset_name='dataset.csv', calculate_stats=True)
            '''
        if save_path is None:
            self.save_path = Path(img_folder).parent
        else:
            self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        df = self.create_location_dataframe(use_labels=label_folder!=None, img_folder=img_folder, 
                                            label_folder=label_folder, img2label_function=img2label_function)

        _ = self.create_training_dataframe(training_split, mask=None, save_name=dataset_name, existing_df=df, save_df=True)

        if calculate_stats:
            if TRANSFORMS is None:
                TRANSFORMS = create_transform(TRAINING_TRANSFORMS_BASIC, mean=[1,1,1], std=[1,1,1]) 
            self.calculate_tile_statistics(transformation=TRANSFORMS, num_of_samples = 100, dataset_name=dataset_name, save_csv=True)


    def create_location_dataframe(self, use_labels=True, img_folder=None, label_folder=None, img2label_function=None):
        """Creates the dataframe listing the path to the x,y tile pairs"""
        if img_folder is None:
            img_folder = os.path.join(self.save_path, self.save_folders[0])
        img_filenames = self._image_files_list(img_folder)

        if use_labels:
            cols = ['x_location', 'y_location', 'dataset_part']
            if  label_folder is None:
                label_folder = os.path.join(self.save_path, self.save_folders[1])
        else:
            cols = ['x_location', 'dataset_part']

        df = pd.DataFrame(index=range(len(img_filenames)), columns=cols)

        for tile_num, tile_name in enumerate(img_filenames):
            df.at[tile_num, "x_location"] = os.path.join(img_folder, tile_name)
            if use_labels:
                if  img2label_function is None:
                    label_tile_name = tile_name
                else:
                    label_tile_name = img2label_function(tile_name)
                df.at[tile_num, "y_location"] = os.path.join(label_folder, label_tile_name)

        return df
    
    def apply_mask(self, df, mask_location):
        """Checks every tile in dataframe and labels it as Masked if under that mask"""

        # load and pad mask
        mask = io.imread(mask_location)
        padding_dims = ImageTiler.calculate_padding_dims(self.tile_size, mask.shape)
        mask = np.pad(mask, ((0, padding_dims[0]), (0, padding_dims[1])), 'reflect')

        # Check if the dataframe has the correct columns
        if "dataset_part" not in df.columns:
            df["dataset_part"] = ""

        # Get coordinates from save string and check if tile is under a mask 
        for i in range(len(df)):
            tile_loc = df.iloc[i]["x_location"]
            y_loc, x_loc = ImageTiler.extract_coordinate(tile_loc)
            
            if self.is_tile_masked(mask, [y_loc, x_loc]):
                df.at[i, "dataset_part"] = "Mask"
        
        return df
    
    def create_training_dataframe(self, training_split, mask=None, save_name="dataset.csv", existing_df=None, save_df=True):
        """Creates dataframe with training split"""

        assert np.sum(training_split).astype(np.float32) == 1.0, f"Error! Sum of split percentages should equal 1, received split {training_split} which sums to: {sum(training_split)}!"

        if existing_df is None:
            df = self.create_location_dataframe()
        else:
            df = existing_df

        if mask is not None:
            df = self.apply_mask(df, mask_location=mask)
        
        # Select only non-masked tiles and shuffle
        non_masked_tiles = df[df["dataset_part"] != "Mask"]

        # Calculate the number of rows in each dataset
        num_of_tiles = len(non_masked_tiles)
        num_training_tiles = round(num_of_tiles * training_split[0])
        num_validation_tiles = round(num_of_tiles * training_split[1])

        # Create a list of indexes for each split
        non_masked_indexes = non_masked_tiles.index
        non_masked_indexes = np.random.permutation(non_masked_indexes)

        training_indexes = non_masked_indexes[0:num_training_tiles]
        validation_indexes = non_masked_indexes[num_training_tiles:num_training_tiles + num_validation_tiles]
        if training_split[2] >0 :
            test_indexes = non_masked_indexes[num_training_tiles + num_validation_tiles:]
        else:
            test_indexes = []

        # Set values in original dataframe
        df.loc[training_indexes, "dataset_part"] = "Train"
        df.loc[validation_indexes, "dataset_part"] = "Validation"
        df.loc[test_indexes, "dataset_part"] = "Test"

        if save_df:
            # # Save results
            save_str = os.path.join(self.save_path, save_name)
            df.to_csv(save_str,index=False)

            json_save_name = os.path.splitext(save_name)[0] + ".json"
            
            self.create_summary_json(df, save_name=json_save_name)

        return df
    
    def create_standard_dataset(self, x_input_path, y_input_path, training_split, mask=False, labels2binary=True):
        """Wrapper class for creating a simple training set"""
        
        self.tile_inputs(x_input_path, y_input_path, labels2binary)

        self.create_training_dataframe(training_split, mask=mask)

    def create_kfold_dataset(self, num_of_folds, test_set_frac, mask=None, save_name="kfold_dataset.csv", existing_df=None):
        
        # First create dataset with standard split. 
        kfold_size = 1 - test_set_frac
        df = self.create_training_dataframe([kfold_size, 0, test_set_frac], mask, save_df=False, existing_df=existing_df)
        df["dataset_part"].replace({"Train": "Kfold"}, inplace=True)

        # Select all indices from the df that are not masked or in the test set
        kfold_tiles = df[df["dataset_part"]=="Kfold"]
        kfold_index = kfold_tiles.index

        # Shuffle indices and split into k chunks
        kfold_index = np.random.permutation(kfold_index)
        kfold_index_split = np.array(np.array_split(kfold_index, num_of_folds), dtype=object)
        
        for i in range(num_of_folds):
            # ith split will be used for validation
            kfold_training = list(range(num_of_folds))
            kfold_training.pop(i)

            validation_tiles = kfold_index_split[i]
            training_tiles = np.hstack(kfold_index_split[kfold_training])
            tt = training_tiles.shape
            # Create a new column for each fold and label rows
            fold_str = "Fold_" + str(i)
            df[fold_str] = df["dataset_part"]

            df.loc[training_tiles, fold_str] = "Train"
            df.loc[validation_tiles, fold_str] = "Validation"
        
        save_str = os.path.join(self.save_path, save_name)
        df.to_csv(save_str, index=False)

        # Create the json summary file
        json_save_name = os.path.splitext(save_name)[0] + ".json"   
        self.create_summary_json(df, save_name=json_save_name)

        return df

    def kfold_dataset_from_folder(self, img_folder, label_folder=None, num_of_folds=3, test_set_frac=0.2, img2label_function=None,
                                  save_path=None, dataset_name='kfold_dataset.csv', calculate_stats=False, TRANSFORMS=None):
        '''Convinence function that handles creation of a k-fold dataset from a folder of images tiles and, optionally a folder of label tiles.
            Args:
                img_folder          str, path to the folder holding the satellite image tiles 
                label_folder        str or None, path to the folder holding the label image tiles[default: None]
                num_of_folds:       int, the number of folds [default: 3]
                test_set_frac:      float (<1.0), the perecentage of data to keep aside as a test set [default: 0.2]
                img2label_function: callable or None, a function that maps the names of the image tiles to the name of the labels tiles,
                                    can be None if the image and labels have the same names [default: None]. 
                                    if supplied the function should consume an input_image_name and yield the correspsongin label_image_name,
                                    e.g. label_tile_name = img2label_function(img_tile_name)
                save_path:          str or None, location of the folder in which the dataset.csv file will be saved [default: None]. 
                dataset_name:       str, the name of the dataset.csv file that will be created [default: 'kfold_dataset.csv']
                calculate_stats:    boolean, if set to True the function will ran the balaning statistics calculation 
                                    This is REQUIRED FOR TRAINING but not for evaluating [default: False] 
                TRANSFORMS:         pytorch transforms, the kind of joint transports that will be used for training (required for calculate_stats)
                                    if None, a set of basic transforms will be used [default: None]
            Returns:
                nothing, saves a dataset.csv file 
            Usage:
                >>> tiler = CNN_tiler(tile_size=512)
                >>> tiler.kfold_dataset_from_folder(img_folder=img_folder, label_folder=label_folder, num_of_folds=5, test_set_frac=0.2, 
                                                  save_path=/path/to/save, dataset_name='5fold_dataset.csv', calculate_stats=True)
            '''
        if save_path is None:
            self.save_path = Path(img_folder).parent
        else:
            self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # First create dataset with standard split. 
        kfold_size = 1 - test_set_frac
        df = self.create_location_dataframe(use_labels=label_folder!=None, img_folder=img_folder, 
                                            label_folder=label_folder, img2label_function=img2label_function)

        df = self.create_training_dataframe([kfold_size, 0, test_set_frac], mask=None, save_name=dataset_name, existing_df=df, save_df=False)
        df["dataset_part"].replace({"Train": "Kfold"}, inplace=True)

        # Select all indices from the df that are not masked or in the test set
        kfold_tiles = df[df["dataset_part"]=="Kfold"]
        kfold_index = kfold_tiles.index

        # Shuffle indices and split into k chunks
        kfold_index = np.random.permutation(kfold_index)
        kfold_index_split = np.array(np.array_split(kfold_index, num_of_folds), dtype=object)
        
        for i in range(num_of_folds):
            # ith split will be used for validation
            kfold_training = list(range(num_of_folds))
            kfold_training.pop(i)

            validation_tiles = kfold_index_split[i]
            training_tiles = np.hstack(kfold_index_split[kfold_training])
            # Create a new column for each fold and label rows
            fold_str = "Fold_" + str(i)
            df[fold_str] = df["dataset_part"]

            df.loc[training_tiles, fold_str] = "Train"
            df.loc[validation_tiles, fold_str] = "Validation"
        
        save_str = os.path.join(self.save_path, dataset_name)
        df.to_csv(save_str, index=False)

        # Create the json summary file
        json_save_name = os.path.splitext(dataset_name)[0] + ".json"
        self.create_summary_json(df, save_name=json_save_name)

        if calculate_stats:
            self.calculate_tile_statistics(transformation=None, num_of_samples = 100, dataset_name=dataset_name, save_csv=True)

        return df


    def create_summary_json(self, df, save_name="dataset.json"):
        """Calculates mean and std of training tiles and outputs result into json alongwith other key informaiton"""
        
        df_train = df[df["dataset_part"].isin(["Train", "Kfold"])].reset_index()
        if len(df_train) > 0:
            # only calculate the mean and std if train set has values
            mean, std, num_channels = CNNTiler.calculate_mean_std(df_train)
        else:
            num_channels = None
            mean, std = np.nan, np.nan
        
        if 'Kfold' in df['dataset_part'].unique():
            val_size = len(df[df["Fold_0"] == "Validation"])
        else:
            val_size = len(df[df["dataset_part"] == "Validation"])
        
        summary = {
            "mean" : mean,
            "std" : std,
            "num_of_channels" : num_channels,
            "original_input_size" : self.input_size,
            "tile_size" : self.tile_size,
            "num_of_tiles" : len(df),
            "num_of_masked_tiles" : len(df[df["dataset_part"] == "Mask"]),
            "training_set_size" : len(df_train),
            "validation_set_size" : val_size,
            "test_set_size" : len(df[df["dataset_part"] == "Test"])
        }
        json.dump(summary, open(os.path.join(self.save_path, save_name) , 'w'), indent=4)

    def is_tile_masked(self, mask, coord):
        ''' checks the mask to see if the current tile is under the mask'''
        mask_tile = mask[coord[0]:coord[0] + self.tile_size, coord[1]:coord[1]+self.tile_size]

        non_zero_values = np.count_nonzero(mask_tile)
        percentage_masked = 1 - (non_zero_values / mask_tile.shape[0] **2)
        return percentage_masked > self.mask_threshold_percentage

    @staticmethod
    def calculate_mean_std(dataset_df):
        """Reset the index of the dataframe before passing it here"""

        # Find the number of channels in the image
        a_tile = np.atleast_3d(io.imread(dataset_df.at[0, "x_location"]))
        channels = a_tile.shape[-1]

        # Calculate the mean for each channel. 
        mean = np.zeros([channels,])
        for i in range(len(dataset_df)):
            img = np.atleast_3d(io.imread(dataset_df.at[i, "x_location"]))
            img = img/256

            for channel in range(channels):
                mean[channel] += img[:,:,channel].mean()

        mean = mean/len(dataset_df)

        # Calculate the variance for each channel
        var = np.zeros([channels,])
        for i in range(len(dataset_df)):
            img = np.atleast_3d(io.imread(dataset_df.at[i, "x_location"]))
            img = img/256

            for channel in range(channels):
                var[channel] += np.square(img - mean[channel]).mean()
            
            #var += np.square(img.numpy() - mean[:,np.newaxis, np.newaxis]).mean(axis=(1,2))

        var = var/len(dataset_df)
        std = np.sqrt(var)

        return mean.tolist(), std.tolist(), channels

    @staticmethod
    def convert_y_labels_to_binary(arr):
        slum_pixel_value = 1
        slum_threshold = 64 #  Predefined value

        return np.where(arr >= slum_threshold, slum_pixel_value, 0).astype("uint8")
    
    # def calculate_tile_statistics(self, transformation=None, num_of_samples = 100, dataset_name="dataset.csv", save_csv=True):

    #     """Updates the training csv with slum coverage statistics for each tile"""

    #     df = pd.read_csv(os.path.join(self.save_path, dataset_name))
    #     df["slum_coverage"] = np.nan
    #     df["average_slum_coverage"] = np.nan
    #     df["slum_sampling_prob"] = np.nan
    #     df["min_coverage"] = np.nan
    #     df["max_coverage"] = np.nan

    #     if transformation is None:
    #         transformation = create_transform(TRAINING_TRANSFORMS_BASIC, mean=[1,1,1], std=[1,1,1]) 
        
    #     for i in tqdm(range(len(df)), desc="Calculating Statistics"):
    #         # We have to use torch vision read_image otherwise the to_tensor function divides the labels by 256

    #         input_x = read_image(df.at[i, "x_location"])
    #         label = read_image(df.at[i, "y_location"])

    #         slum_coverage = label.sum() / (max(label.shape)**2)
    #         df.at[i, "slum_coverage"] = slum_coverage
    #         if slum_coverage > 0.001:

    #             slum_count = 0
    #             avg_slum_perc = 0
    #             min_coverage = 1.0
    #             max_coverage = 0.0
    #             for j in range(num_of_samples):
    #                 x, y = transformation(input_x, label)
                    
    #                 #  images are in tensor format [C, H, W], labels in [1, H, W]
    #                 slum_percentage = float(y.sum() / (max(y.shape) **2))
    #                 avg_slum_perc += slum_percentage
                    
    #                 if slum_percentage > 0.01:
    #                     slum_count += 1
                    
    #                 if slum_percentage > max_coverage:
    #                     max_coverage = slum_percentage
                    
    #                 if slum_percentage < min_coverage and slum_percentage > 0:
    #                     min_coverage = slum_percentage
                
                
    #             df.at[i, "slum_sampling_prob"] = slum_count/num_of_samples
    #             df.at[i, "average_slum_coverage"] = avg_slum_perc/num_of_samples
    #             df.at[i, "min_coverage"] = min_coverage
    #             df.at[i, "max_coverage"] = max_coverage

    #     if save_csv:
    #         df.to_csv(os.path.join(self.save_path, dataset_name), index=False)
    #     else:
    #         return df

############################ Development 

    def calculate_tile_statistics(self, transformation=None, num_of_samples = 100, dataset_name="dataset.csv", save_csv=True):

        """Updates the training csv with slum coverage statistics for each tile"""

        df = pd.read_csv(os.path.join(self.save_path, dataset_name))
        df["slum_coverage"] = np.nan
        df["average_slum_coverage"] = np.nan
        df["slum_sampling_prob"] = np.nan
        df["min_coverage"] = np.nan
        df["max_coverage"] = np.nan
        
        for i in tqdm(range(len(df)), desc="Calculating Statistics"):
            # We have to use torch vision read_image otherwise the to_tensor function divides the labels by 256

            input_x = read_image(df.at[i, "x_location"])
            label = read_image(df.at[i, "y_location"])
            # if isinstance(label, torch.Tensor):
            #     print(f"label is a PyTorch tensor with dtype: {label.dtype}")


            if i <=1:
                img_size = np.prod(label.shape)
                max_val = label.max()
                if max_val > 1.: # working with signed distances
                    threshold = 64 
                else: # working with binary labels
                    threshold = 0.5
                if transformation is None:
                    transformation = TRAINING_TRANSFORMS_BASIC
                    transformation['joint_transforms'].insert(0, BinarizeLabels())
                    transformation = create_transform(transformation, mean=[0,0,0], std=[1,1,1]) 

            slum_coverage = (label >= threshold).sum() / img_size
            # print(f'Debug info - img_size: {img_size.dtype}')
            # print(f'Debug info - slum_coverage: {slum_coverage}')
            # print(f'Debug info - slum_coverage.dtype: {slum_coverage.dtype}')
            df.at[i, "slum_coverage"] = slum_coverage.item() # set to item

            if slum_coverage > 0.001:

                slum_count = 0
                avg_slum_perc = 0
                min_coverage = 1.0
                max_coverage = 0.0
                for j in range(num_of_samples):
                    x, y = transformation(input_x, label)
                    
                    #  images are in tensor format [C, H, W], labels in [1, H, W]
                    slum_percentage = float(y.sum() / (max(y.shape) **2))
                    avg_slum_perc += slum_percentage
                    
                    if slum_percentage > 0.01:
                        slum_count += 1
                    
                    if slum_percentage > max_coverage:
                        max_coverage = slum_percentage
                    
                    if slum_percentage < min_coverage and slum_percentage > 0:
                        min_coverage = slum_percentage
                
                
                df.at[i, "slum_sampling_prob"] = slum_count/num_of_samples
                df.at[i, "average_slum_coverage"] = avg_slum_perc/num_of_samples
                df.at[i, "min_coverage"] = min_coverage
                df.at[i, "max_coverage"] = max_coverage

        if save_csv:
            df.to_csv(os.path.join(self.save_path, dataset_name), index=False)
        else:
            return df

#############################################################################################


    @staticmethod
    def trim_image(img_path, size, save_name):

        img = io.imread(img_path)
        
        if len(img.shape) == 3:
            img = img[0:size[0], 0:size[1], :]
        else:
            img = img[0:size[0], 0:size[1]]
        
        io.imsave(save_name, img, check_contrast=False)

    def create_inference_dataset(self, x_input_path, mask=None):
        """Creates the dataframe for inference training"""

        tiler = ImageTiler(x_input_path, self.tile_size)
        tiler.tile_and_save(self.save_path)

        df = self.create_location_dataframe(use_labels=False)

        if mask is not None:
            df = self.apply_mask(df, mask_location=mask)
        
        df[df["dataset_part"] != "Mask"] = "Inference"

        df.to_csv(os.path.join(self.save_path, "inference_dataset.csv"))
        
        return df

def combine_datasets(csv_location_list, save_location):
    """Combines multiple tiled datasets into one. Recalculates statistics creates new Json file
            Args:
            csv_location_list:          file, list of dataset.csv locations 
            save_location:              str, full path (and filename) to the location that the combined dataset and json will be saved
        USAGE:
        >>> from cnn_tiler import combine_datasets
        >>> combine_datasets(['/path/to/dataset1/dataset.csv',...,'/path/to/datasetn/dataset.csv'], '/path/to/location/to/save/combined_dataset.csv)
    """
    os.makedirs(os.path.dirname(save_location),exist_ok=True)
    save_location = Path(save_location)
    save_location = str(save_location.parent/save_location.stem) # remove the file suffix
    # load all the csvs
    df_list = []
    for csv_loc in csv_location_list:
        if csv_loc.endswith('.csv'):
            df_list.append(pd.read_csv(csv_loc))
    
    concat_df = pd.concat(df_list, ignore_index=True)

    train_df = concat_df[concat_df["dataset_part"] == "Train"].reset_index()
    if len(train_df) > 0:
        # only calculate the mean and std if train set has values
        mean, std, num_channels = CNNTiler.calculate_mean_std(train_df)
    else:
        num_channels = None
        mean, std = np.nan, np.nan

    # write mean/std to json
    summary = {
        "mean" : mean,
        "std" : std,
        "num_of_channels" : num_channels,
        "num_of_tiles" : len(concat_df),
        "num_of_masked_tiles": len(concat_df[concat_df["dataset_part"] == "Mask"]),
        "training_set_size" : len(concat_df[concat_df["dataset_part"] == "Train"]),
        "validation_set_size" : len(concat_df[concat_df["dataset_part"] == "Validation"]),
        "test_set_size" : len(concat_df[concat_df["dataset_part"] == "Test"])
    }
    with open(save_location+".json", 'w') as fout:
        json.dump(summary, fout, indent=4)
    with open(save_location+".csv", 'w') as fout:
        concat_df.to_csv(fout, index=False)


def resplit_dataset(dataset_csv, split_ratio=[0.6,0.2,0.2], save_location=None):
    """Loads an existing dataset.csv file and produces a different train-val-test split also recalculating normalization statistics.
       Creates new csv and json files.
            Args:
            dataset_csv:                str, existing dataset.csv file
            split_ratio:                str, train-val-test split [default:[0.6,0.2,0.2]]
            save_location:              str, full path (and filename) to the location that the combined dataset and json will be saved
                                        if left empty the script will overwrite the existing files [default:None]
        USAGE:
        >>> from cnn_tiler import resplit_dataset
        >>> resplit_dataset(['/path/to/dataset/dataset.csv', split_ratio=[0.8,0.2,0.0])
    """
    assert (dataset_csv.endswith('.csv') and os.path.exists(dataset_csv)), f"Could not fine the csv file provided: {dataset_csv}"
    assert sum(split_ratio) == 1.0, f"Sum of split ratio should sum to 1.0. Supplie values {split_ratio} do not."
    if save_location is not None: 
        os.makedirs(os.path.dirname(save_location),exist_ok=True)
    else:
        save_location = dataset_csv
    save_location = Path(save_location)
    save_location = str(save_location.parent/save_location.stem) # remove the file suffix
    df = pd.read_csv(dataset_csv)
    
    non_masked_indices = df.index[df["dataset_part"] != 'Mask']
    num_tiles = len(non_masked_indices)
    non_masked_indices = np.random.permutation(non_masked_indices)

    train_ids = non_masked_indices[: int(split_ratio[0] * num_tiles)]
    val_ids = non_masked_indices[int(split_ratio[0] * num_tiles):int(sum(split_ratio[:2]) * num_tiles)]
    test_ids = non_masked_indices[int(sum(split_ratio[:2]) * num_tiles):]

    df.loc[train_ids, "dataset_part"] = "Train"
    df.loc[val_ids, "dataset_part"] = "Validation"
    df.loc[test_ids, "dataset_part"] = 'Test' 

    train_df = df[df["dataset_part"] == "Train"].reset_index()
    if len(train_df) > 0:
        # only calculate the mean and std if train set has values
        mean, std, num_channels = CNNTiler.calculate_mean_std(train_df)
    else:
        num_channels = None
        mean, std = np.nan, np.nan

    # write mean/std to json
    summary = {
        "mean" : mean,
        "std" : std,
        "num_of_channels" : num_channels,
        "num_of_tiles" : len(df),
        "num_of_masked_tiles": len(df[df["dataset_part"] == "Mask"]),
        "training_set_size" : len(df[df["dataset_part"] == "Train"]),
        "validation_set_size" : len(df[df["dataset_part"] == "Validation"]),
        "test_set_size" : len(df[df["dataset_part"] == "Test"])
    }
    with open(save_location+".json", 'w') as fout:
        json.dump(summary, fout, indent=4)
    with open(save_location+".csv", 'w') as fout:
        df.to_csv(fout, index=False)
