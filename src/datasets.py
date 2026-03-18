import os
import torch
import pandas as pd

from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.transforms import Compose, Resize
try:
      import slumworldML.src.custom_transformations as ct
except Exception as Error:
      try:
            import custom_transformations as ct
      except Exception as Error:
            import src.custom_transformations as ct


class TrainDataSet(torch.utils.data.Dataset):

      def __init__(self, training_df, transform=False):
            self.df = training_df
            self.transform = transform
      
      def __len__(self):
            'Denotes the total number of samples'
            return len(self.df)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            filename_x = self.df.iloc[index]["x_location"]
            filename_y = self.df.iloc[index]["y_location"]

            save_name = filename_x.split(os.sep)[-1]

            # Load each image
            x = read_image(filename_x).float()
            y = read_image(filename_y).float()

            if self.transform is not False:
                  x, y = self.transform(x, y)

            return x, y, save_name

class ValidationDataset(torch.utils.data.Dataset):

      def __init__(self, training_df, transform=False, original_size=512, new_size=256):
            self.df = training_df
            self.transform = transform
            self.new_size = new_size
            self.num_of_tiles = int(original_size/new_size)**2

            # create new coordinates [[y1, x1], [y1, x2], .., [y2, x1]]
            coords = [i*new_size for i in range(int(original_size/new_size))]
            self.top_left_point = []
            for height in coords:
                  for width in coords:
                        self.top_left_point.append([height, width])
            
      
      def create_save_name(self, filename, y_loc, x_loc):
            coor = filename.split(os.sep)[-1]
            coor = os.path.splitext(coor)[0]
            coor = coor.split("_")
            y_loc_tile = int(coor[0]) + y_loc
            x_loc_tile = int(coor[1]) + x_loc

            return str(y_loc_tile) + "_" + str(x_loc_tile) + ".png"
            
      def __len__(self):
            'Denotes the total number of samples'
            return len(self.df)*self.num_of_tiles

      def __getitem__(self, index):
            
            df_index = int(index/self.num_of_tiles)   # int acts as floor operation
            tile_number = index%self.num_of_tiles     # remainder from division to choose tile

            # get the height and width values for the current sub-tile
            y_loc = self.top_left_point[tile_number][0]
            x_loc = self.top_left_point[tile_number][1]

            'Generates one sample of data'
            # Select sample
            filename_x = self.df.iloc[df_index]["x_location"]
            filename_y = self.df.iloc[df_index]["y_location"]

            # Load each image
            x = read_image(filename_x).float()
            y = read_image(filename_y).float()

            if self.transform is not False:
                  x, y = self.transform(x, y)
            
            x_ = x[:, y_loc:y_loc+self.new_size, x_loc:x_loc+self.new_size]
            y_ = y[:, y_loc:y_loc+self.new_size, x_loc:x_loc+self.new_size]

            save_name = self.create_save_name(filename_x, y_loc, x_loc)
            

            return x_, y_, save_name


class InferenceDataset(torch.utils.data.Dataset):
      
      def __init__(self, tiled_folder_path, transform=False):
            # get all items from the folder
            # all .png files
            self.tile_filenames = os.listdir(tiled_folder_path)
            self.tiled_folder_path = tiled_folder_path

            self.transform = transform

      
      def __len__(self):
            'Denotes the total number of samples'
            return len(self.tile_filenames)
      
      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            filename_x = self.tile_filenames[index]

            # Load each image
            x = read_image(os.path.join(self.tiled_folder_path, filename_x)).float()

            if self.transform is not False:
                  x = self.transform(x)

                  # loaded images are [0, 256] range. Need to get them into [0, 1]
                  #x = x/256

            return x, filename_x


if __name__ == "__main__":

      from torch.utils.data import DataLoader

      ds = SatelliteDataset("/home/raza/code/slumworldML/data/round_75_Briana_MUL_inputs/MD/TiledImage_Paddedtilesize_512Tilesize_256", "train", True)

      train_dataloader = DataLoader(ds, batch_size=2)
      x, y = next(iter(train_dataloader))

      print(x.shape)
      print(y.shape)
