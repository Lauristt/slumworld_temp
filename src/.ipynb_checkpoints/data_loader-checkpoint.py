import os
import random

import torch
import pandas as pd

from torchvision import transforms
from torchvision.io import read_image



class SatiliteDataset(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, tiled_folder, mode, tile_size, transform):
            """ Creates dataset needed for PyTorch dataloader

            """
            self.data_folder = tiled_folder
            self.df = pd.read_csv(tiled_folder + "/info.csv")
            self.mode = mode
            self.tile_size = tile_size
            self.transform = transform

            self.jitter = transforms.ColorJitter(brightness=.5, hue=.5, contrast=0.5, saturation=0.5)
            self.sharpness_adjuster = transforms.RandomAdjustSharpness(sharpness_factor=2)

            # TODO make sure mode fits into one of these

            # Balance the training set

            if mode == "train":
                  self.df = self.df.loc[self.df['training_set'] == "Train"]
            if mode == "validation":
                  self.df = self.df.loc[self.df['training_set'] == "Validation"]
            if mode == "test":
                  self.df = self.df.loc[self.df['training_set'] == "Test"]

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.df)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            filename_x = self.df.iloc[index]["x_location"]
            filename_y = self.df.iloc[index]["y_location"]

            # Load each image
            x = read_image(os.path.join(self.data_folder, filename_x))
            y = read_image(os.path.join(self.data_folder, filename_y))

            x = x/256

            # Have to perform the trasformation mannually on both images.
            if self.transform:
                  
                  # Randomly get tile size and then use random crop and then resize to correct tile size.

                  # randomly crop image
                  i, j, h, w = transforms.RandomCrop.get_params(
                                    x, output_size=(self.tile_size, self.tile_size))
                  x = transforms.functional.crop(x, i, j, h, w)
                  y = transforms.functional.crop(y, i, j, h, w)

                  # random horizontal flip
                  if random.random() > 0.5:
                        x = transforms.functional.hflip(x)
                        y = transforms.functional.hflip(y)
                  
                  # random vertical flip
                  if random.random() > 0.5:
                        x = transforms.functional.vflip(x)
                        y = transforms.functional.vflip(y)
                  
                  # Random colour jitter can only be performed on images with 3 channels.
                  if random.random() > 0.5 and x.shape[0] == 3:

                        x = self.jitter(x)
                  
                  # Randomly adjust sharpness
                  if random.random() > 0.5:
                        x = self.sharpness_adjuster(x)
                  
            return x, y


if __name__ == "__main__":

      from torch.utils.data import DataLoader

      ds = SatiliteDataset("/home/raza/code/slumworldML/data/round_73_Briana_PAN_inputs/MD_python/TiledImage_Paddedtilesize_512Tilesize_256", "train", 256, True)

      train_dataloader = DataLoader(ds, batch_size=2)
      x, y = next(iter(train_dataloader))
