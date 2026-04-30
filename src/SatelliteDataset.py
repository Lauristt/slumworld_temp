import sys
import os
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from typing import Optional, List, Dict, Any
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import  LightningDataModule
from packaging import version
import torch.nn.functional as F
from torchvision import transforms
import requests
from tqdm import tqdm
from pytorch_lightning.trainer.supporters import CombinedLoader
# from lightning.pytorch.utilities import CombinedLoader


try:
    from slumworldML.src.transforms_loader import ( TRAINING_TRANSFORMS_PAN, TRAINING_TRANSFORMS, 
                                                    INFERENCE_TRANSFORMS, create_transform, INFERENCE_TRANSFORMS_DICT)
    from slumworldML.src.custom_transformations import RandomRotateZoomCropTensor, LabelNoise
except Exception as Error:
    try:
        from src.custom_transformations import RandomRotateZoomCropTensor, LabelNoise
        from src.transforms_loader import ( TRAINING_TRANSFORMS_PAN, TRAINING_TRANSFORMS, 
                                            INFERENCE_TRANSFORMS, create_transform, INFERENCE_TRANSFORMS_DICT)
    except Exception as Error:
        from custom_transformations import  RandomRotateZoomCropTensor, LabelNoise
        from transforms_loader import ( TRAINING_TRANSFORMS_PAN, TRAINING_TRANSFORMS, 
                                        INFERENCE_TRANSFORMS, create_transform, INFERENCE_TRANSFORMS_DICT)

import pdb


def get_dinov3_features_from_disk(feature_path):
    try:
        return torch.load(feature_path)
    except FileNotFoundError:
        print(f"Warning: Feature file not found at {feature_path}. Using image only.")
        return None

class BaseCNNDataset(torch.utils.data.Dataset):
    # This is the base class for our datasets.
    def __init__(self):
        super().__init__()
        # Default DINOv3 parameters, will be overridden by child classes.
        self.use_dinov3_features = False
        self.dino_features_path = None
        self.dino_feature_dim = 1024
        self.dino_patch_size = 16

    @staticmethod
    def create_save_name(filename, y_loc, x_loc):
        # Creates a unique filename for a cropped tile.
        coor = filename.split(os.sep)[-1]
        coor = os.path.splitext(coor)[0]
        coor = coor.split("_")
        y_loc_tile = int(coor[0]) + y_loc
        x_loc_tile = int(coor[1]) + x_loc
        return str(y_loc_tile) + "_" + str(x_loc_tile) + ".png"

    def _make_dummy_dino_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a dummy DINO feature map with spatial size consistent with the image size and patch size.
        This prevents slicing (for sub-tiles) from producing 0-sized tensors when DINO is disabled.
        """
        img_h, img_w = x.shape[-2:]

        # Use floor division to match typical patch embedding grid size.
        feat_h = max(1, img_h // self.dino_patch_size)
        feat_w = max(1, img_w // self.dino_patch_size)

        # Keep dtype/device consistent with x
        return torch.zeros(
            (1, feat_h, feat_w),
            dtype=x.dtype,
            device=x.device,
        )

    def _get_x_y_filename(self, index):
        # Core logic to get an image, its label, and its DINOv3 feature.
        if hasattr(self, 'df_other') and self.df_other is not None and not self.df_other.empty:
            if np.random.choice([0, 1, 2]) > 0:
                index = index % (self.orsize)
                filename_x = self.df.iloc[index]["x_location"]
                filename_y = self.df.iloc[index]["y_location"]
            else:
                index = np.random.choice(len(self.df_other))
                filename_x = self.df_other.iloc[index]["x_location"]
                filename_y = self.df_other.iloc[index]["y_location"]
        else:
            index = index % (self.orsize)
            filename_x = self.df.iloc[index]["x_location"]
            filename_y = self.df.iloc[index]["y_location"]

        # Load image and label from disk or RAM.
        if self.from_disc:
            x = read_image(filename_x)
            y = read_image(filename_y)
        else:
            x = self.read_image_from_disc(filename_x, islabel=False)
            y = self.read_image_from_disc(filename_y, islabel=True)
        
        # Apply transformations if they exist.
        if self.transform is not False:
            x, y = self.transform(x, y)
        
        if self.use_dinov3_features:
            base_name = os.path.basename(filename_x).replace('.png', '.pt').replace('.jpg', '.pt')
            feature_path = os.path.join(self.dino_features_path, base_name)
            dino_features = get_dinov3_features_from_disk(feature_path)

            # If feature file is missing, create a correctly-sized zero tensor dynamically.
            if dino_features is None:
                img_h, img_w = x.shape[-2:]
                feat_h = max(1, img_h // self.dino_patch_size)
                feat_w = max(2, img_w // self.dino_patch_size)
                print(
                    f"Warning: DINOv3 feature not found for {base_name}. "
                    f"Creating a zero tensor of size ({self.dino_feature_dim}, {feat_h}, {feat_w})."
                )
                dino_features = torch.zeros(
                (self.dino_feature_dim, feat_h, feat_w),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            # If not using DINOv3, create a placeholder tensor.
            dino_features = self._make_dummy_dino_features(x)

        # Return all items as separate tensors.
        return x, dino_features, y, filename_x

    def _get_x_filename(self, index):
        # This method is for inference when no label (y) is present.
        index = index % (self.orsize - 1)
        filename_x = self.df.iloc[index]["x_location"]

        if self.from_disc:
            x = read_image(filename_x)
        else:
            x = self.read_image_from_disc(filename_x, islabel=False)

        if self.transform is not False:
            x = self.transform(x, None)

        if self.use_dinov3_features:
            base_name = os.path.basename(filename_x).replace('.png', '.pt').replace('.jpg', '.pt')
            feature_path = os.path.join(self.dino_features_path, base_name)
            dino_features = get_dinov3_features_from_disk(feature_path)

            if dino_features is None:
                img_h, img_w = x.shape[-2:]
                feat_h = max(1, img_h // self.dino_patch_size)
                feat_w = max(1, img_w // self.dino_patch_size)
                print(
                    f"Warning: DINOv3 feature not found for {base_name}. "
                    f"Creating a zero tensor of size ({self.dino_feature_dim}, {feat_h}, {feat_w})."
                )
                dino_features = torch.zeros(
                    (self.dino_feature_dim, feat_h, feat_w),
                    dtype=x.dtype,
                    device=x.device,
                )
        else:
            dino_features = self._make_dummy_dino_features(x)

        return x, dino_features, filename_x


    def read_image_from_disc(self, filename, islabel=False):
        # Reads an image from RAM if pre-loaded.
        if islabel:
            out = copy.deepcopy(self.y_dict[filename])
        else:
            out = copy.deepcopy(self.x_dict[filename])
        return out

    def _load_to_ram(self, df):
        # Pre-loads the entire dataset into RAM.
        for file in df.x_location:
            self.x_dict[file] = read_image(file)
        try:
            for file in df.y_location:
                self.y_dict[file] = read_image(file)
        except Exception as Error:
            pass

class TrainDataSet(BaseCNNDataset):
    def __init__(self, balancing_df, complement_df=None, transform=False, tile_size=None, 
                 size_mutliplier=10, in_ram_dataset=False, 
                 use_dinov3_features=False, dino_features_path=None,dino_feature_dim=1024, dino_patch_size=16):
        super().__init__()
        self.df = balancing_df
        self.df_other = complement_df
        self.transform = transform
        self.orsize = len(self.df)
        self.from_disc = not in_ram_dataset 
        self.x_dict = {}
        self.y_dict = {}
        
        # Store DINOv3 parameters.
        self.use_dinov3_features = use_dinov3_features
        self.dino_features_path = dino_features_path
        self.dino_feature_dim = dino_feature_dim 
        self.dino_patch_size = dino_patch_size

        if in_ram_dataset:
            self._load_to_ram(self.df)
            if self.df_other is not None:
                self._load_to_ram(self.df_other)

        # Determine if tiles should be split into smaller sub-tiles.
        transform_types = [str(type(t)) for t in transform.joint_transforms]
        if str(type(RandomRotateZoomCropTensor())) in transform_types:
            self.SPLIT_TILE = False
            self.size_multiplier = size_mutliplier
        else:
            self.SPLIT_TILE = True
            original_size = tile_size if tile_size is not None else 512
            self.subtile_size = int(original_size / 2)
            self.tiles_split_multiplier = (original_size // self.subtile_size) ** 2
            self.size_multiplier = size_mutliplier if size_mutliplier >= 4 else self.tiles_split_multiplier
            coords = [i * self.subtile_size for i in range(original_size // self.subtile_size)]
            self.top_left_point = [[h, w] for h in coords for w in coords]

    def __len__(self):
        return len(self.df) * self.size_multiplier 

    def __getitem__(self, index):
        # Get all separate tensors from the base class method.
        # if split tile then randomly select tiles from the [2*2] split tiles
        x, dino_features, y, filename_x = self._get_x_y_filename(index)
        
        if not self.SPLIT_TILE:
            save_name = filename_x.split(os.sep)[-1]
            return x, dino_features, y, save_name
        else:
            # If splitting tiles, crop all tensors (image, dino_features, label).
            tile_number = np.random.choice(list(range(self.tiles_split_multiplier)))
            y_loc, x_loc = self.top_left_point[tile_number]
            
            x_ = x[:, y_loc:y_loc + self.subtile_size, x_loc:x_loc + self.subtile_size]
            y_ = y[:, y_loc:y_loc + self.subtile_size, x_loc:x_loc + self.subtile_size]
            
            # Crop DINOv3 features proportionally. This is a crucial step.
            sub_dino_h = self.subtile_size // self.dino_patch_size
            sub_dino_w = self.subtile_size // self.dino_patch_size
            dino_y_loc = y_loc // self.dino_patch_size
            dino_x_loc = x_loc // self.dino_patch_size
            dino_features_ = dino_features[:, dino_y_loc:dino_y_loc + sub_dino_h, dino_x_loc:dino_x_loc + sub_dino_w]
            
            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            return x_, dino_features_, y_, save_name

class ValidationDataset(BaseCNNDataset):
    def __init__(self, validation_df, transform=False, tile_size=None, 
                 split_tiles=True, mode='eval',
                 use_dinov3_features=False, dino_features_path=None,
                 dino_feature_dim=1024, dino_patch_size=16):
        super().__init__()
        self.df = validation_df
        self.orsize = len(self.df) # Added orsize for consistency
        self.transform = transform
        self.from_disc = True
        self.eval_mode = (mode == 'eval') or (mode == 'evaluation')
        self.x_dict = {} # Initialize dicts even if not used, for safety
        self.y_dict = {}

        # Store DINOv3 parameters.
        self.use_dinov3_features = use_dinov3_features
        self.dino_features_path = dino_features_path
        self.dino_feature_dim = dino_feature_dim 
        self.dino_patch_size = dino_patch_size 
        
        original_size = tile_size if tile_size is not None else 512
        self.split_tiles = split_tiles
        
        if split_tiles:
            self.subtile_size = int(original_size / 2)
            self.tiles_split_multiplier = (original_size // self.subtile_size) ** 2
            coords = [i * self.subtile_size for i in range(original_size // self.subtile_size)]
            self.top_left_point = [[h, w] for h in coords for w in coords]
        else: 
            self.tiles_split_multiplier = 1

    def __len__(self):
        return len(self.df) * self.tiles_split_multiplier

    def __getitem__(self, index):
        df_index = int(index / self.tiles_split_multiplier)
        
        # Use the base class method to get the full-size tensors.
        # Note: We call _get_x_y_filename which handles loading x, y, and dino_features.
        x, dino_features, y, filename_x = self._get_x_y_filename(df_index)

        if self.split_tiles:
            tile_number = index % self.tiles_split_multiplier
            y_loc, x_loc = self.top_left_point[tile_number]
            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            
            x_ = x[:, y_loc:y_loc + self.subtile_size, x_loc:x_loc + self.subtile_size]
            
            sub_dino_h = self.subtile_size // self.dino_patch_size
            sub_dino_w = self.subtile_size // self.dino_patch_size
            dino_y_loc = y_loc // self.dino_patch_size
            dino_x_loc = x_loc // self.dino_patch_size
            dino_features_ = dino_features[:, dino_y_loc:dino_y_loc + sub_dino_h, dino_x_loc:dino_x_loc + sub_dino_w]
            if self.eval_mode:
                y_ = y[:, y_loc:y_loc + self.subtile_size, x_loc:x_loc + self.subtile_size]
                return x_, dino_features_, y_, save_name
            else:
                y_ = torch.zeros_like(y[:, y_loc:y_loc+self.subtile_size,
                         x_loc:x_loc+self.subtile_size])
                return x_, dino_features_,y_, save_name
        else:
            save_name = filename_x.split(os.sep)[-1]
            y_ = torch.zeros_like(y)
            if self.eval_mode:
                return x, dino_features, y, save_name
            else:
                y_ = torch.zeros_like(y)
                return x, dino_features, y_, save_name

class InferenceDataset(torch.utils.data.Dataset):
    # This class is for inference on a folder of images.
    def __init__(self, tiled_folder_path, transform=False, 
                 use_dinov3_features=False, dino_features_path=None,
                 dino_feature_dim=1024, dino_patch_size=16):
        super().__init__()
        self.tile_filenames = [f for f in os.listdir(tiled_folder_path) if f.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']]
        self.tiled_folder_path = tiled_folder_path
        self.transform = transform

        # Store DINOv3 parameters.
        self.use_dinov3_features = use_dinov3_features
        self.dino_features_path = dino_features_path
        self.dino_feature_dim = dino_feature_dim
        self.dino_patch_size = dino_patch_size
    
    def __len__(self):
        return len(self.tile_filenames)
    
    def __getitem__(self, index):
        filename_x = self.tile_filenames[index]
        image_path = os.path.join(self.tiled_folder_path, filename_x)
        x = read_image(image_path)

        if self.transform is not False:
                x = self.transform(x)
        
        dino_features = None
        if self.use_dinov3_features:
            # Assumes feature files are in a parallel directory.
            base_name = os.path.basename(filename_x).replace('.png', '.pt').replace('.jpg', '.pt')
            feature_path = os.path.join(self.dino_features_path, base_name)
            dino_features = get_dinov3_features_from_disk(feature_path)
            
            if dino_features is None:
                img_h, img_w = x.shape[-2:]
                feat_h = img_h // self.dino_patch_size
                feat_w = img_w // self.dino_patch_size
                dino_features = torch.zeros(self.dino_feature_dim, feat_h, feat_w)
        else:
            dino_features = torch.zeros(1, 1, 1)

        # Return image, DINO features, and the filename.
        return x, dino_features, filename_x

# new dev, implement dinov3 featrues
# new dev, support gradient-supervised (one-shot learning) and not gradient-supervised (domain adaptation task)
class AdaptationDataset(BaseCNNDataset):
    def __init__(self, adaptation_df,
                 transform=False,
                 tile_size=None,
                 size_mutliplier=4,
                 in_ram_dataset=False,
                 supervised = False,
                 use_dinov3_features=False,
                 dino_features_path=None,
                 dino_feature_dim = 1024,
                 dino_patch_size = 16
                 ):
        super().__init__()
        self.df = adaptation_df
        self.transform = transform
        self.orsize = len(self.df)
        self.from_disc = not in_ram_dataset
        self.supervised = supervised # if return label
        self.x_dict = {}
        self.y_dict = {}

        self.use_dinov3_features = use_dinov3_features
        self.dino_features_path = dino_features_path
        self.dino_feature_dim = dino_feature_dim
        self.dino_patch_size = dino_patch_size

        if in_ram_dataset:
            self._load_to_ram(self.df)
        transform_types = [str(type(t)) for t in transform.joint_transforms]
        if str(type(RandomRotateZoomCropTensor())) in transform_types:
            # we do not need to split the tile it will be handled inside the transforms
            self.SPLIT_TILE = False
            self.tiles_split_multiplier = 1
            self.size_multiplier = size_mutliplier
        else:
            self.SPLIT_TILE = True
            if tile_size is None:
                print("Tile size not provided. Assuming an input tile_size of 512.")
                original_size = 512
            else:
                original_size = tile_size
            self.subtile_size = int(original_size/2)
            self.tiles_split_multiplier = int(original_size/self.subtile_size)**2
            self.size_multiplier = size_mutliplier if size_mutliplier >=4 else self.tiles_split_multiplier
            # create new coordinates [[y1, x1], [y1, x2], .., [y2, x1]]
            coords = [i*self.subtile_size for i in range(int(original_size/self.subtile_size))]
            self.top_left_point = []
            for height in coords:
                for width in coords:
                        self.top_left_point.append([height, width])

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.df)*self.size_multiplier

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.supervised:
            x, dino_features, y, filename_x = self._get_x_y_filename(index)
        else:
            x, dino_features, filename_x = self._get_x_filename(index)
            y = torch.zeros_like(x[0:1])

        if not self.SPLIT_TILE:
            '''Generates one sample of data when the crop_split_resize transform is applied hence
                there is no need to split the tile into subtiles'''
            save_name = filename_x.split(os.sep)[-1]
            return x, dino_features, y, save_name
        else:
            # select one of the possible subtiles
            tile_number = np.random.choice(list(range(self.tiles_split_multiplier)))
            # get the height and width values for the current sub-tile
            y_loc = self.top_left_point[tile_number][0]
            x_loc = self.top_left_point[tile_number][1]
            x_ = x[:, y_loc:y_loc+self.subtile_size , x_loc:x_loc+self.subtile_size]
            sub_dino_h = self.subtile_size // self.dino_patch_size
            sub_dino_w = self.subtile_size // self.dino_patch_size
            dino_y_loc = y_loc // self.dino_patch_size
            dino_x_loc = x_loc // self.dino_patch_size
            dino_features_ = dino_features[:, dino_y_loc:dino_y_loc + sub_dino_h, dino_x_loc:dino_x_loc + sub_dino_w]
            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            y_ = y[:, y_loc:y_loc + self.subtile_size, x_loc:x_loc + self.subtile_size]
            return x_, dino_features_, y_, save_name

class SatelliteDataModule(LightningDataModule):
    def __init__(self, dataset_file, adaption_task_dataset_file=None, required_slum_coverage=0.00001, 
                 size_mutliplier=8, norm_file=None, test_datapath=None, train_batch_size=4, 
                 test_batch_size=10, num_workers=4, shuffle=True, tile_size=None, foldID=-1, 
                 image_type='mul', overfit_mode=False, ssp=False, mask_sample_fraction=0.5, 
                 label_noise=False, in_ram_dataset=False, training_transformations=None, 
                 validation_transformations=None, use_dinov3_features=False, dino_features_path=None,
                 dino_feature_dim=1024, dino_patch_size=16, finetuning_dataset_file = None, finetuning_config = None,
                 domain_mix_mode='loader_level',target_per_batch = 1):
        super().__init__()
        self.use_dinov3_features = use_dinov3_features
        self.dino_features_path = dino_features_path 
        self.dino_feature_dim = dino_feature_dim 
        self.dino_patch_size = dino_patch_size

        self.finetuning_dataset = finetuning_dataset_file
        self.with_finetuning = finetuning_dataset_file is not None
        self.finetuning_config = finetuning_config or {}

        self.mix_mode = self.finetuning_config.get('mix_mode',domain_mix_mode)
        self.target_samples_cfg = self.finetuning_config.get('samples_per_batch',target_per_batch)

        if self.with_finetuning:
            try:
                self.ft_df = pd.read_csv(self.finetuning_dataset)
                self.ft_df.fillna(0.0, inplace=True)
                self.ft_df_train = self.ft_df
                print(f"[SatelliteDataModule] Loaded Fine-tuning dataset: {len(self.ft_df)} samples.")
            except Exception as e:
                print(f"Error loading fine-tuning CSV: {e}. Please set target_finetuning-enabled to be false if you do not want to do low-shot finetuning.")
                sys.exit(1)

        self.dataset_file = dataset_file
        self.adaption_dataset_file = adaption_task_dataset_file
        self.with_adaption_task = adaption_task_dataset_file is not None
        self.test_datapath = test_datapath
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.slum_coverage = required_slum_coverage
        self.size_mutliplier = size_mutliplier
        self.tile_size = tile_size
        self.foldID = foldID
        self.overfit_mode = overfit_mode
        self.PAN_IMAGE = image_type == 'pan'
        self.image_type = image_type
        self.MASK_SAMPLE_FRACTION = mask_sample_fraction
        self.label_noise = label_noise
        self.in_ram_dataset = in_ram_dataset
        self.training_transformations = training_transformations
        self.validation_transformations = validation_transformations
        self.mode = ssp if ssp is not None else 'train'
        self.df = pd.read_csv(self.dataset_file)
        self.adf = pd.read_csv(self.adaption_dataset_file) if self.with_adaption_task else None


        if norm_file is not None:
            with open(norm_file, 'r') as json_file:
                data = json.load(json_file)
            self.mean = data["mean"]
            self.std = data["std"]
            if len(self.mean) == 1: self.PAN_IMAGE = True
        else:
            sys.exit("Normalization File not Provided. Exiting...")
        self.setup(stage=None)

    def setup(self, stage=None):
        training_column = f"Fold_{self.foldID}" if isinstance(self.foldID, int) and self.foldID != -1 else "dataset_part"
        self.df.fillna(0.0, inplace=True)
        self.df_train = self.df[self.df[training_column] == "Train"]
        self.df_mask_sample = self.df[self.df[training_column] == "Mask"].sample(frac=self.MASK_SAMPLE_FRACTION)
        self.df_train_other = pd.concat([self.df_train[self.df_train["average_slum_coverage"] < self.slum_coverage]]*2 + [self.df_mask_sample]).reset_index(drop=True)
        self.df_train = self.df_train[self.df_train["average_slum_coverage"] >= self.slum_coverage]
        self.df_validation = self.df[self.df[training_column] == "Validation"]
        self.df_test = self.df[self.df[training_column] == "Test"]
        
        if self.with_adaption_task:
            self.adf.fillna(0.0, inplace=True)
            self.adf_train = self.adf[self.adf[training_column].isin(['Train', 'Mask'])]
            self.adf_validation = self.adf[self.adf[training_column] == 'Validation'] if not self.adf[self.adf[training_column] == 'Validation'].empty else self.adf_train

        TRAIN_TRANSFORMS = self.training_transformations or (TRAINING_TRANSFORMS_PAN if self.PAN_IMAGE else TRAINING_TRANSFORMS)
        INFER_TRANSFORMS = self.validation_transformations or INFERENCE_TRANSFORMS
        
        complement_df = None if self.overfit_mode else self.df_train_other
        if self.label_noise:
            TRAIN_TRANSFORMS['joint_transforms'].append(LabelNoise(noise_level=0.1))

        train_transforms = create_transform(TRAIN_TRANSFORMS, mean=self.mean, std=self.std, input_size=self.tile_size//2)
        infer_transforms = create_transform(INFER_TRANSFORMS, mean=self.mean, std=self.std, input_size=self.tile_size//2)
    
        if stage in ('train', None):
            self.train_dataset = TrainDataSet(
                balancing_df=self.df_train, 
                complement_df=complement_df, 
                transform=train_transforms, 
                tile_size=self.tile_size, 
                size_mutliplier=self.size_mutliplier, 
                in_ram_dataset=self.in_ram_dataset, 
                use_dinov3_features=self.use_dinov3_features, 
                dino_features_path=self.dino_features_path,
                dino_feature_dim=self.dino_feature_dim, 
                dino_patch_size=self.dino_patch_size
            )
            self.validation_dataset = ValidationDataset(
                validation_df=self.df_validation, 
                transform=infer_transforms, 
                tile_size=self.tile_size, 
                use_dinov3_features=self.use_dinov3_features, 
                dino_features_path=self.dino_features_path,
                dino_feature_dim=self.dino_feature_dim, 
                dino_patch_size=self.dino_patch_size
            )
            self.test_dataset = ValidationDataset(
                validation_df=self.df_test, 
                transform=infer_transforms, 
                tile_size=self.tile_size,
                use_dinov3_features=self.use_dinov3_features,
                dino_features_path=self.dino_features_path
            )
            if self.with_adaption_task:
                self.adaptation_train_dataset = AdaptationDataset(
                    adaptation_df=self.adf_train, 
                    transform=copy.deepcopy(train_transforms), 
                    tile_size=self.tile_size, 
                    in_ram_dataset=self.in_ram_dataset,
                    supervised= False,
                    use_dinov3_features=self.use_dinov3_features,
                    dino_features_path=self.dino_features_path,
                    dino_feature_dim=self.dino_feature_dim,
                    dino_patch_size=self.dino_patch_size
                )
                self.adaptation_validation_dataset = AdaptationDataset(
                    adaptation_df=self.adf_validation, 
                    transform=infer_transforms, 
                    tile_size=self.tile_size,
                    supervised = False,
                    use_dinov3_features=self.use_dinov3_features,
                    dino_features_path=self.dino_features_path,
                    dino_feature_dim=self.dino_feature_dim,
                    dino_patch_size=self.dino_patch_size
                )
            if self.with_finetuning:
                ft_dino_path = self.finetuning_config.get('target_dino_feature_path', self.dino_features_path)
                self.finetuning_train_dataset = AdaptationDataset(
                    adaptation_df=self.ft_df_train,
                    transform=copy.deepcopy(train_transforms),
                    tile_size=self.tile_size,
                    in_ram_dataset=self.in_ram_dataset,
                    supervised=True,
                    use_dinov3_features=self.use_dinov3_features,
                    dino_features_path=ft_dino_path,
                    dino_feature_dim=self.dino_feature_dim,
                    dino_patch_size=self.dino_patch_size
                )
                self.finetuning_val_dataset = AdaptationDataset(
                    adaptation_df=self.ft_df_train,
                    transform=infer_transforms,
                    tile_size=self.tile_size,
                    supervised=True,
                    use_dinov3_features=self.use_dinov3_features,
                    dino_features_path=ft_dino_path,
                    dino_feature_dim=self.dino_feature_dim,
                    dino_patch_size=self.dino_patch_size
                )

            self.tiles_per_epoch = len(self.train_dataset)
        elif stage == "test":
             self.test_dataset = InferenceDataset(
                tiled_folder_path=self.test_datapath, 
                transform=infer_transforms, 
                use_dinov3_features=self.use_dinov3_features,
                dino_features_path=self.dino_features_path
            )


    # def train_dataloader(self):
    #     dataloaders_dict = {}
    #     train_loader_workers = self.num_workers if self.num_workers > 0 else 0
    #     train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=train_loader_workers, persistent_workers=train_loader_workers > 0, shuffle=self.shuffle, pin_memory=True)
    #     dataloaders_dict['slum_prediction'] = train_dataloader
    #     if self.with_adaption_task:
    #         adaptation_dataloader = DataLoader(self.adaptation_train_dataset, batch_size=self.train_batch_size, num_workers=train_loader_workers, persistent_workers=train_loader_workers > 0, shuffle=self.shuffle, pin_memory=True)
    #         dataloaders_dict['domain_prediction'] = adaptation_dataloader
    #     return CombinedLoader(dataloaders_dict, mode='max_size_cycle')
    def train_dataloader(self):
        dataloaders_dict = {}
        train_loader_workers = self.num_workers if self.num_workers > 0 else 0
        # loader-level

        source_bs = self.train_batch_size
        ft_bs = 0 # finetune batch size
        should_drop_last = False

        # batch-level
        
        if self.with_finetuning and (self.mix_mode == 'in_batch'):
            ft_dataset_len = len(self.finetuning_train_dataset) if hasattr(self.finetuning_train_dataset, '__len__') else len(self.ft_df_train)
            if self.target_samples_cfg == -1:
                ft_bs = ft_dataset_len
                print(f"[Fine-Tuning] 'samples_per_batch' is -1. Using ALL available samples: {ft_bs}")
            else:
                ft_bs = int(self.target_samples_cfg)
                if ft_bs > ft_dataset_len:
                    # this is to avoid empty fine-tuning dataloader
                    print(
                        f"Warning: Configured samples_per_batch ({ft_bs}) is larger than dataset size ({ft_dataset_len}).")
                    print(f"Resulting loader would be empty with drop_last=True. Adjusting ft_bs to {ft_dataset_len}.")
                    ft_bs = ft_dataset_len

            if ft_bs >= self.train_batch_size:
                raise ValueError(f"Domain Adaptation Config Error: "
                                 f"Target batch size ({ft_bs}) must be strictly less than "
                                 f"total batch_size ({self.train_batch_size}).\n"
                                 f"  -> If target_per_batch is -1, your target dataset is too large for this batch size.\n"
                                 f"  -> Your current source size is {len(self.train_dataset)}"
                                 f"  -> Please increase 'batch_size' or reduce target data usage.")

            if ft_bs < 1:
                print(f"Warning: Calculated target batch size is {ft_bs}. Disabling in-batch mixing... Fallback to loader-level mixing")
                ft_bs = 0

            if ft_bs > 0:
                # total source = batch size - finetune batch size
                source_bs = self.train_batch_size - ft_bs
                should_drop_last = True

                print(f"\n[Fine-Tuning] Strategy: 'in_batch' mixing enabled.")
                print(f"  -> Total Batch Size: {self.train_batch_size}")
                print(f"  -> Source Samples (Slum): {source_bs}")
                print(f"  -> Target Samples (New Domain): {ft_bs}")

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=source_bs,
                                      num_workers=train_loader_workers,
                                      persistent_workers=train_loader_workers > 0,
                                      shuffle=self.shuffle,
                                      pin_memory=True,
                                      drop_last=should_drop_last
                                      )
        dataloaders_dict['slum_prediction'] = train_dataloader

        if self.with_finetuning and ft_bs>0:
            finetuning_dataloader = DataLoader(
                self.finetuning_train_dataset,
                batch_size=ft_bs,
                num_workers=train_loader_workers,
                persistent_workers=train_loader_workers > 0,
                shuffle=True,
                pin_memory=True,
                drop_last=should_drop_last
                # should also discuss empty batch case here.. mark this for later dev
            )
            dataloaders_dict['target_finetuning'] = finetuning_dataloader
        elif self.with_finetuning and self.mix_mode != 'in_batch':
            finetuning_dataloader = DataLoader(
                self.finetuning_train_dataset,
                batch_size=self.train_batch_size,
                num_workers=train_loader_workers,
                shuffle=True,
                pin_memory=True
            )
            dataloaders_dict['target_finetuning'] = finetuning_dataloader

        if self.with_adaption_task:
            adaptation_dataloader = DataLoader(self.adaptation_train_dataset,
                                               batch_size=self.train_batch_size,
                                               num_workers=train_loader_workers,
                                               persistent_workers=train_loader_workers > 0,
                                               shuffle=self.shuffle,
                                               pin_memory=True,
                                               drop_last=should_drop_last)
            dataloaders_dict['domain_prediction'] = adaptation_dataloader
        return CombinedLoader(dataloaders_dict, mode='max_size_cycle')

    def val_dataloader(self):
        val_loader_workers = self.num_workers if self.num_workers > 0 else 0
        dataloaders_dict = {}

        #main validation
        dataloaders_dict['slum_prediction'] = DataLoader(
            self.validation_dataset,
            batch_size=self.test_batch_size,
            num_workers=val_loader_workers,
            persistent_workers=val_loader_workers > 0,
            shuffle=False,
            pin_memory=True
        )

        #fine-tuning Validation
        if self.with_finetuning:
            dataloaders_dict['target_finetuning'] = DataLoader(
                self.finetuning_val_dataset,
                batch_size=self.test_batch_size,
                num_workers=val_loader_workers,
                persistent_workers=val_loader_workers > 0,
                shuffle=False,
                pin_memory=True
            )

        #domain-adaptation
        if self.with_adaption_task:
            dataloaders_dict['domain_prediction'] = DataLoader(
                self.adaptation_validation_dataset,
                batch_size=self.test_batch_size,
                num_workers=val_loader_workers,
                persistent_workers=val_loader_workers > 0,
                shuffle=False,
                pin_memory=True
            )

        if len(dataloaders_dict) == 1:
            return [dataloaders_dict['slum_prediction']]

        return [CombinedLoader(dataloaders_dict, mode='max_size_cycle')]

    def test_dataloader(self):
        test_loader_workers = self.num_workers if self.num_workers > 0 else 0
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=test_loader_workers, shuffle=False, pin_memory=True)
        return test_dataloader

    def predict_dataloader(self):
        return self.test_dataloader()

class InferenceDataLoader:
    # This is a simplified wrapper for creating an inference dataloader.
    def __init__(self, datapath, norm_file=None, batch_size=25, num_workers=4, shuffle=False,
                 use_dinov3_features=False, dino_features_path=None,
                 dino_feature_dim=1024, dino_patch_size=16):
        super().__init__()
        if norm_file is not None:
            with open(norm_file, 'r') as json_file:
                data = json.load(json_file)
            self.mean = data["mean"]
            self.std = data["std"]
        else:
            self.mean, self.std = [0.0], [1.0]
        
        # It does NOT load a DINOv3 model anymore.
        
        infer_transforms = create_transform(INFERENCE_TRANSFORMS, mean=self.mean, std=self.std)
        
        # It instantiates the refactored InferenceDataset with the correct parameters.
        self.dataset = InferenceDataset(
            tiled_folder_path=datapath, 
            transform=infer_transforms, 
            use_dinov3_features=use_dinov3_features,
            dino_features_path=dino_features_path,
            dino_feature_dim=dino_feature_dim,
            dino_patch_size=dino_patch_size
        )
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    def get_dataloader(self):
        return self.dataloader


class TestingDataLoader:
    ''' DataLoader to be used for Evaluation/Testing
    Loads all tiles from a tiled folder, normalizes and presents them to the model.
    ARGS:
        dataset_file:   str, full path to the dataset.csv file [default: None]
        norm_file       str, full path to the normalization file used during training [default: None]
        batch_size      int, batch size [default: 10]
        num_workers     int, number of workers to use for loading and preprocessing images [default: 2]
        tile_size:      int, [default: 512]
        shuffle:        boolean, shuffle data [default: False]
        foldID:         int, for k-fold CV, -1 for standard training [default: -1]
        split_tiles     boolean, split tiles to [tile_size/2, tile_size/2], the tile_size seen by the network [default:True]
        TTA:            boolean, load Test Time Augmentation transformations [default:False]
        image_type:     str, one of 'mul', 'pan' [default: 'mul']
        mode:           str, one of 'eval', 'infer' [default: 'eval'], if set to infer
                        the dataloader will only contain images and tile_names (not labels)
        use_only_test_tiles: boolean, [default:False]
                        if set to False it will return all the data from the dataset.csv (i.e. 'train'/'val'/'test' splits)
                        if set to True it will only return the data from 'test' split 
    Usage:
        # >>> experiment = "/home/minas/slumworld/data/output/experiments/SatelliteUnet/lightning_logs/version_18/"
        # >>> checkpoint = experiment + "checkpoints/BinaryCrossEntropyWithLogits-AdamW--L2_2e-05-Seed_1357-TStamp_1629139845-epoch=225-val_loss=0.0606-val_acc=0.9757-val_f1=0.7871808409690857.ckpt'
        # >>> dataset_file = "/home/minas/slumworld/data/tiled/MD_MUL_75_Briana/tiled_input/dataset.csv"
        # >>> Loader = TestingDataLoader(dataset_file, batch_size=25, num_workers=4, tile_size=512, norm_file=norm_file)
        # >>> testDataLoader = Loader.get_dataloader()
    '''
    # This is a simplified wrapper for creating a testing dataloader from a CSV file.
    def __init__(self, dataset_file, norm_file=None, batch_size=10, num_workers=1, tile_size=512,
                 shuffle=False, foldID=-1, split_tiles=True, TTA=False, image_type='mul',
                 mode='eval', use_only_test_tiles=False, exclude_test_tiles=False,
                 use_dinov3_features=False, dino_features_path=None,
                 dino_feature_dim=1024, dino_patch_size=16):

        df = pd.read_csv(dataset_file)
        training_column = f"Fold_{foldID}" if isinstance(foldID, int) and foldID != -1 else "dataset_part"
        if use_only_test_tiles:
            self.df = df[df[training_column] == "Test"]
            if self.df.empty:
                sys.exit("Error: No 'Test' data found for the specified fold.")
        elif exclude_test_tiles:
            self.df = df[df[training_column] != "Test"]
            if self.df.empty:
                sys.exit("Error: No non-Test data found after excluding Test tiles.")
        else:
            self.df = df
            
        if norm_file:
            with open(norm_file, 'r') as f: data = json.load(f)
            self.mean, self.std = data["mean"], data["std"]
        else:
            self.mean, self.std = [0.0], [1.0]

        # It does NOT load a DINOv3 model anymore.

        transform_key = 'tta_' + image_type if TTA else 'inference_mul'
        transforms_config = INFERENCE_TRANSFORMS_DICT.get(transform_key, INFERENCE_TRANSFORMS)
        infer_transforms = create_transform(transforms_config, mean=self.mean, std=self.std)
        
        # It instantiates the refactored ValidationDataset with the correct parameters.
        self.dataset = ValidationDataset(
            validation_df=self.df, 
            transform=infer_transforms, 
            tile_size=tile_size, 
            split_tiles=split_tiles, 
            mode=mode, 
            use_dinov3_features=use_dinov3_features,
            dino_features_path=dino_features_path,
            dino_feature_dim=dino_feature_dim,
            dino_patch_size=dino_patch_size
        )
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)

    def get_dataloader(self):
        return self.dataloader
