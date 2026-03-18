"""
Module is for the custom image crop and rotation logic designed to be used with pytourch data loader
"""
import random
import copy
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F
import torch
from skimage import filters
import pdb

class SegmentationCompose:
    """Modified pytorch compose class to accept image and label for combining transformations.
    
    Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, input_transformation, joint_transformation, input_tile_size, normalization_stats=None):
        self.input_transforms = input_transformation
        self.joint_transforms = joint_transformation
        self.input_tile_size = input_tile_size
        self.normalize = transforms.Normalize(mean=normalization_stats[0], std=normalization_stats[1])
        

    def __call__(self, img, label=None):
        if len(self.input_transforms) > 0:
            # apply joint transformations
            for t in self.input_transforms:
                # print("transform", t)
                # print("Max of image:",img.max())
                img = t(img)
        if len(self.joint_transforms) > 0:
            if label is not None:
                # apply joint transformations
                for t in self.joint_transforms:
                    if isinstance(t, Resize):
                        img, label = t(input=img, label=label, size=self.input_tile_size) #, antialias=None)
                    elif isinstance(t, SSP_Generator) or isinstance(t, Autoencoding_SSP_Generator):
                        # we need the actual tile-size not the CNN input size since it is at the start of transforms
                        img, label = t(img, label, input_tile_size=2*self.input_tile_size)
                    else:
                        img, label = t(img, label) 
            else:
                # apply joint transformations
                for t in self.joint_transforms:
                    if isinstance(t, Resize):
                        img  = t(input=img, size=self.input_tile_size) #, antialias=None)
                    elif isinstance(t, SSP_Generator) or isinstance(t, Autoencoding_SSP_Generator):
                        # we need the actual tile-size not the CNN input size since it is at the start of transforms
                        img = t(img, input_tile_size=2*self.input_tile_size)
                    else:
                        # print(t)
                        img = t(img) 
        else:
            img = img.type(torch.FloatTensor)
        # if self.normalize is not None:
        #     img = self.normalize(img)
        if label is not None:
            return img, label
        else:
            return img


class split_tile(object):
    def __init__(self, original_size=512, new_size=None, num_of_channels=3):
        self.original_size = original_size

        if new_size is not None:
            self.new_size = new_size
        else:
            self.new_size = int(original_size/2)

        self.num_of_channels = num_of_channels

        # preallocate new tiles
        num_of_tiles = int(original_size/new_size)**2
        self.stacked_x = torch.empty((num_of_tiles, self.num_of_channels, new_size, new_size))
        self.stacked_y = torch.empty((num_of_tiles, 1, new_size, new_size))
        
        # create new coordinates [[y1, x1], [y1, x2], .., [y2, x1]]
        coords = [i*new_size for i in range(int(original_size/new_size))]
        self.top_left = []
        for height in coords:
            for width in coords:
                self.top_left.append([height, width])

    def __call__(self, x, y=None):
        for i, coords in enumerate(self.top_left):
            y1 = coords[0]
            x1 = coords[1]
            self.stacked_x[i, :, :, :] = x[:, y1:y1+self.new_size, x1:x1+self.new_size]
            if y is not None:
                self.stacked_y[i, :, :, :] = y[:, y1:y1+self.new_size, x1:x1+self.new_size]
        if y is not None:
            return self.stacked_x, self.stacked_y
        else:
            return self.stacked_x

class RandomChannelShuffle(object):
    def __init__(self, shuffle_probability=0.5):
        self.shuffle_prob = shuffle_probability
        
    def __call__(self, input):
        if random.random() < self.shuffle_prob:
            new_order = torch.randperm(3).tolist()
            res = torch.empty_like(input)
            res = input[new_order,:,:]
            return res
        else:
            return input

class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, input, label=None):
        if label is not None:
            return self.to_tensor(input), self.to_tensor(label)
        else:
            return self.to_tensor(input)

# class TrivialAugmentBasic(object):
#     def __init__(self, ):
#         self.transform = TrivialAugment()
    
#     def __call__(self, input, label):
#         x,y = self.transform(input, label)
#         return x, y

class LabelNoise(object):
    def __init__(self, noise_level=0.1):
        assert noise_level<1.0, f"Label noise level should be smaller than 1.0. Supplied value is {noise_level}. Aborting ...."
        self.noise_level = noise_level
    
    def __call__(self, input, label=None):
        if label is not None:
            return input, torch.clamp(label.float(), min=self.noise_level, max=1-self.noise_level)
        else:
            return input

class LabelNoiseFromDistances(object):
    def __init__(self, noise_level=0.1, slum_pixel_value = 1, slum_threshold = 64):
        assert noise_level<1.0, f"Label noise level should be smaller than 1.0. Supplied value is {noise_level}. Aborting ...."
        self.noise_level = noise_level
        self.slum_threshold = slum_threshold
    
    def __call__(self, input, label=None):
        if label is not None:
            lbl = torch.where(label>=self.slum_threshold, 1-self.noise_level, self.noise_level)
            lbl[label==0] = 0.0
            return input, lbl
        else:
            return input

class BinarizeLabels(object):
    def __init__(self, slum_pixel_value = 1, slum_threshold = 64):
        ''' Dyamically binarize a sign distance based slum map 
        ARGS:
            slum_threshold    int, value used for slum boundaries 
                              (smaller values corresponds to points inside slums, higher values outside slums)
            slum_pixel_value  int, class label for slums (default: 1)
        '''
        self.slum_pixel_value = slum_pixel_value
        self.slum_threshold = slum_threshold
    
    def __call__(self, input, label=None):
        if label is not None:
            return input, torch.where(label >= self.slum_threshold, self.slum_pixel_value, 0).to(torch.uint8)
        else:
            return input

class ProbWrapper(object):
    def __init__(self, prob=1.0, transformation=None):
        self.p = prob
        self.transformation = transformation

    def __call__(self, input, label=None):
        if (self.transformation is not None) and (np.random.choice(a=[0,1], p =(1-self.p, self.p))):
                if label is None:
                    return self.transformation.__call__(input)
                else:
                    return self.transformation.__call__(input, label)
        else:
            if label is None:
                return input
            else:
                return input, label

class Resize(object):
    def __init__(self):
        # self.size = (size, size)
        # self.label_size = (size, size)
        pass
    
    def __call__(self, input, label=None, size=None):
        
        size_tuple = (size, size)

        x = torch.unsqueeze(input, 0).float()
        x = F.interpolate(x, size=size_tuple, mode='bilinear', align_corners=False)
        x = torch.squeeze(x, 0)
        if label is not None:
            y = torch.unsqueeze(label, 0).float()
            y = F.interpolate(y, size=size_tuple, mode='bilinear', align_corners=False)
            y = torch.squeeze(y, 0)
            return x, y
        else:
            return x

class ZeroToOneRange(object):
    def __call__(self, input, label=None, upper_value=255):
        if label is None:
            return torch.true_divide(input, upper_value)
        else:
            return torch.true_divide(input, upper_value), label

class RandomHflip(object):
    def __call__(self, input, label=None):
        if random.random() > 0.5:
            if label is not None:
                return transforms.functional.hflip(input), transforms.functional.hflip(label)
            else: 
                return transforms.functional.hflip(input)
        else:
            if label is not None:
                return input, label
            else:
                return input

class RandomVflip(object):
    def __call__(self, input, label=None):
        if random.random() > 0.5:
            if label is not None:
                return transforms.functional.vflip(input), transforms.functional.vflip(label)
            else:
                return transforms.functional.vflip(input)
        else:
            if label is not None:
                return input, label
            else:
                return input

class RandomRotateZoomCropTensor(object):
    """
    Randomly rotates an image and randomly crops a tile of random size. Output tile does not include and padding from rotation.
    
    image and label must pytorch image tensor
    """

    def __init__(self, min_tile_fraction=0.5):
        self.angles = np.arange(91)
        self.min_tile_fraction = min_tile_fraction
        
    
    def get_square(self, image, label=None):
        
        padded_tile_size = image.shape[-1]
        self.min_output_size = int(padded_tile_size*self.min_tile_fraction)

        # Get a random angle
        angle = np.deg2rad(np.random.choice(self.angles))

        # get largest square that can fit into the tiled square whilst rotated by sampled angle
        largest_square_length = padded_tile_size / (np.cos(angle) + np.sin(angle)) -1

        # sample the new tile size
        rotated_tile_length = np.random.randint(low=self.min_output_size, high=int(largest_square_length))

        # calculate the area from which we will sample the top left corner of the rotated tile
        top_left = np.sin(angle) * rotated_tile_length
        top_left = int(np.ceil(top_left))

        bottom_left = padded_tile_size - (np.cos(angle) * rotated_tile_length + top_left)
        bottom_left = int(np.floor(bottom_left))

        top_right = padded_tile_size - np.cos(angle) * rotated_tile_length
        top_right = int(np.floor(top_right))

        # if top_left >= top_right:
        #     return image, label
            
        # sample a point from square
        y_value = np.random.randint(bottom_left)
        x_value = np.random.randint(low=top_left, high=top_right)

        # rotate the image and label
        rotated_img = transforms.functional.rotate(image, np.rad2deg(angle), expand=True)
        if label is not None:
            rotated_label = transforms.functional.rotate(label, np.rad2deg(angle), expand=True)
        #rotated_img = ndimage.rotate(image, np.rad2deg(angle))
        #rotated_label = ndimage.rotate(label, np.rad2deg(angle))

        # get the center
        y_center = int(image.shape[-2]/2)
        x_center = int(image.shape[-1]/2)

        # get the new point in terms of the center
        y_value_centered = y_center - y_value
        x_value_centered = x_value - x_center

        # Rotate the point within the new coordinate system
        y_val_rotated = y_value_centered * np.cos(angle) + x_value_centered * np.sin(angle)
        x_val_rotated = x_value_centered * np.cos(angle) - y_value_centered * np.sin(angle)

        # Convert the point into the coordinate system of rotated tile
        y_center_rotated = int(rotated_img.shape[-2]/2)
        x_center_rotated = int(rotated_img.shape[-1]/2)

        y_value_new = int(y_center_rotated - y_val_rotated)
        x_value_new = int(x_center_rotated + x_val_rotated)

        cropped_img = rotated_img[:, y_value_new:y_value_new + rotated_tile_length, x_value_new:x_value_new+rotated_tile_length]
        if label is not None:
            cropped_label = rotated_label[:, y_value_new:y_value_new + rotated_tile_length, x_value_new:x_value_new+rotated_tile_length]
            return cropped_img, cropped_label
        else:
            return cropped_img
 
    
    def __call__(self, image, label=None):
        
        if label is not None:
            try: 
                x, y = self.get_square(image, label)
                return x, y
            except ValueError:
                return image, label
        else:
            try: 
                x = self.get_square(image)
                return x
            except ValueError:
                return image



class SSP_Generator(object):
    """
    Masks a percentage of the image pixels by applying the given RGB color value. Uses MASK global variable.
    
    """

    def __init__(self, threshold=0.05, mask_percentage=15, n_masks=5):
        self.threshold = threshold
        self.mask_percentage = mask_percentage
        self.n_masks = n_masks
        self.counter = 0

    def prepare_masks(self, input_tile_size=None):
        '''Create random masks to be used to mask an input image and use it for self-supervised pre-training.
        ARGS:
            tile_size:          int, the shape of the tiles [default: 512]
            mask_percentage:    int, percentage to mask [default: 15]
            n_masks:            int, number of random masks to prepare [default: 5]
                and (optionaly) the mask percentage
        Returns:
            populates the global variable MASK which will be accessed by the dataloader 
        '''
        assert input_tile_size is not None, "Error! Input_size not provided to SSP call. Cannot create masks!"
        shape = (input_tile_size, input_tile_size)
        total_pixels = np.prod(shape)
        mask_pixels = int(round(total_pixels * self.mask_percentage / 100))
        self.MASK = [np.random.choice(np.arange(total_pixels), size=mask_pixels, replace=False) for _ in range(self.n_masks)]
        self.MASK = np.vstack(self.MASK)

    @staticmethod
    def tosobel(img, threshold=0.1):
        edge_sobel2 = filters.sobel(np.array(img.data)[:,:,0])
        return (edge_sobel2>threshold).astype(np.uint8)[np.newaxis,:,:]

    def __call__(self, image, label_in=None, input_tile_size=None):
        if self.counter % (self.n_masks * 400) == 0:
            # every 400 x n_masks images renew the masks
            self.prepare_masks(input_tile_size)
        mask = self.MASK[np.random.choice(len(self.MASK))]
        if image.is_contiguous():
            image_ = copy.deepcopy(image).view(-1,3)
        else:
            image_ = copy.deepcopy(image).contiguous().view(-1,3)
        shape = image.shape
        image_[mask]= torch.zeros(3,dtype=image_.dtype)
        image_ = image_.view(shape[::-1])
        label = torch.from_numpy(self.tosobel(image_, threshold=self.threshold))
        image_ = image_.view(shape)
        return image_, label


class Autoencoding_SSP_Generator(object):
    """
    Masks a percentage of the image pixels by applying the given RGB color value. Uses MASK global variable.
    
    """

    def __init__(self, threshold=0.05, mask_percentage=15, n_masks=5):
        self.threshold = threshold
        self.mask_percentage = mask_percentage
        self.n_masks = n_masks
        self.counter = 0

    def prepare_masks(self, input_tile_size=None):
        '''Create random masks to be used to mask an input image and use it for self-supervised pre-training.
        ARGS:
            tile_size:          int, the shape of the tiles [default: 512]
            mask_percentage:    int, percentage to mask [default: 15]
            n_masks:            int, number of random masks to prepare [default: 5]
                and (optionaly) the mask percentage
        Returns:
            populates the global variable MASK which will be accessed by the dataloader 
        '''
        assert input_tile_size is not None, "Error! Input_size not provided to SSP call. Cannot create masks!"
        shape = (input_tile_size, input_tile_size)
        total_pixels = np.prod(shape)
        mask_pixels = int(round(total_pixels * self.mask_percentage / 100))
        self.MASK = [np.random.choice(np.arange(total_pixels), size=mask_pixels, replace=False) for _ in range(self.n_masks)]
        self.MASK = np.vstack(self.MASK)

    def __call__(self, image, label_in=None, input_tile_size=None):
        if self.counter % (self.n_masks * 400) == 0:
            # every 400 x n_masks images renew the masks
            self.prepare_masks(input_tile_size)
        mask = self.MASK[np.random.choice(len(self.MASK))]
        if image.is_contiguous():
            image_ = copy.deepcopy(image).view(-1,3)
        else:
            image_ = copy.deepcopy(image).contiguous().view(-1,3)
        shape = image.shape
        label = torch.zeros((image_.shape[0],1))
        threshold = 0.5*(abs(image.max()) - abs(image.min()))
        # Transform RGB to Grayscale: Y = 0.2125 R + 0.7154 G + 0.0721 B
        label[mask] = ((0.2125*image_[mask].squeeze(1)[:,0] + 0.7154 *image_[mask].squeeze(1)[:,1] + 0.0721 *image_[mask].squeeze(1)[:,2])>threshold).float().reshape([-1,1])
        image_[mask] = torch.zeros(3,dtype=image_.dtype)
        image_ = image_.view(shape)
        label = label.reshape([1,shape[-2],shape[-1]]) #.type(torch.uint8)
        return image_, label