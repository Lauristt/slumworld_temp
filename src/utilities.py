import os
import sys
import math
import warnings
import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch import Tensor
from tabulate import tabulate
from matplotlib import pyplot as plt
from functools import wraps
from pathlib import Path
from termios import FF1
import copy
import pdb
from skimage import io
import cv2
import imageio
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


# Returns a double Convolution, for use in the UNet model
def double_conv(in_channels, out_channels, kernel_size=3, use_batch_norm=True, in_place=True):
    
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1), # With padding=1, output height and width remain unchanged
        nn.ReLU(inplace=in_place),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.ReLU(inplace=in_place)
    ]
    if use_batch_norm:
        layers.insert(1, nn.BatchNorm2d(out_channels))
        layers.insert(4, nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

# Get the VGG encoder stages
def get_vgg_encoder_stages(encoder):
    stages = []
    stage_modules = []
    for module in encoder:
        if isinstance(module, nn.MaxPool2d):
            stages.append(nn.Sequential(*stage_modules))
            stage_modules = []
        stage_modules.append(module)
    stages.append(nn.Sequential(*stage_modules))
    return stages

def init_conv2d(layer_, nonlinearity='relu'):
    '''Correctly initialize conv2D layers using kaiming_uniform (pytorch default) 
    settting the activation to the one actually used (here relu) versus the default (leaky_relu)'''
    # https://github.com/pytorch/pytorch/blob/029a968212b018192cb6fc64075e68db9985c86a/torch/nn/modules/conv.py#L49
    if type(layer_) == nn.Conv2d:
        nn.init.kaiming_uniform_(layer_.weight, a=math.sqrt(5), nonlinearity=nonlinearity) 
        if layer_.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer_.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer_.bias, -bound, bound)

# Converts outputs to numpy array
def convert_outputs(outputs):

    if torch.is_tensor(outputs):
        return outputs.numpy()
    else:
        return outputs

# Turns labels to ints and converts them to numpy array
def convert_labels(labels):
    
    if torch.is_tensor(labels):
        labels = labels.long()
        return labels.numpy()
    else:
        return labels


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


def normalise_2D(img):
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)    
    if len(img.shape)>2:
        img = img[:,:,0]
    return img

def save_2D(img, filename):    
    if len(img.shape)>2:
        img = img[:,:,0]
    cv2.imwrite(filename, img)

def crf(original_image, annotated_image, output_image, 
         kernel_size=5, compat=3, colour_kernel_size=80, colour_compat=10,
         n_steps=10, use_2d=True):
    '''
        Original_image = Image which has to labelled
        Annotated image = Which has been labelled by some technique( FCN in this case)
        Output_image = Name of the final output image after applying CRF
        Use_2d = boolean variable 
        if use_2d = True specialised 2D fucntions will be applied
        else Generic functions will be applied
    '''
    # Converting annotated image to RGB if it is Gray scale
    original_annotated_shape = annotated_image.shape
    if(len(original_annotated_shape)<3):
        annotated_image = gray2rgb(annotated_image).astype(np.uint32)
    
    cv2.imwrite("testing2.png", annotated_image)
    annotated_image = annotated_image.astype(np.uint32)
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0].astype(np.uint32) + \
                        (annotated_image[:,:,1]<<8).astype(np.uint32) + \
                            (annotated_image[:,:,2]<<16).astype(np.uint32)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.90, zero_unsure=False) ## You can set zero_unsure=True
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(kernel_size, kernel_size), 
                              compat=compat, 
                              kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC) # original (3, 3)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(colour_kernel_size, colour_kernel_size), 
                               srgb=(13, 13, 13), rgbim=original_image,
                               compat=colour_compat,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(n_steps)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:] / 255
    MAP = MAP.reshape(original_image.shape).astype(np.uint8)

    # MAP = normalise_2D(MAP)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    imageio.imsave(output_image, MAP, format='png')   # save with imageio.imsave
        # img = cv2.imread(output_image, cv2.IMREAD_GRAYSCALE)
        # save_2D(img, output_image)
    return MAP

def compute_micro_metrics(true, pred):
    '''Given a list of true (pixel level) labels and predicted (pixel level) labels 
    returns the micro (i.e. per tile) metrics.
        args:
            true:                   list[np.array], list of tensors that stored the true labels.
            pred:                   list[np.array], list of numpy arrays that store the model predictions.
        returns:
            confusion_matrix:       np.array, dimension [num_batches, batch_size, 4]
            accuracy:               np.array, dimension [num_batches, batch_size]
            precision:              np.array, dimension [num_batches, batch_size]
            recall:                 np.array, dimension [num_batches, batch_size]
            f_one:                  np.array, dimension [num_batches, batch_size]
            iou:                    np.array, dimension [num_batches, batch_size]
    '''
    SMOOTH = 1e-6
    pred = convert_outputs(pred)
    true = convert_labels(true)
    try:
        if pred[-1].shape != pred[-2].shape:    # last batch is smaller
            diff = np.abs(np.array(pred[-2].shape) - np.array(pred[-1].shape)).max() # find difference in batch size
            basic_pad_size = [1]
            basic_pad_size.extend(list(pred[-1].shape[1:]))
            pred[-1] = np.concatenate((pred[-1],np.repeat(np.zeros(basic_pad_size),diff, axis=0))) # pad with zeros
            true[-1] = np.concatenate((true[-1],np.repeat(np.zeros(basic_pad_size),diff, axis=0))) # pad with zeros
    except IndexError:                          #there is only one batch
        pass

    tp = np.logical_and(np.stack(true), np.stack(pred)).squeeze().sum(axis=(-2,-1))
    fp = np.logical_and(np.stack(pred), np.logical_not(np.stack(true))).squeeze().sum(axis=(-2,-1))
    fn = np.logical_and(np.logical_not(np.stack(pred)), np.stack(true)).squeeze().sum(axis=(-2,-1))
    tn = np.logical_and(np.logical_not(np.stack(true)), np.logical_not(np.stack(pred))).squeeze().sum(axis=(-2,-1))
    with np.errstate(divide='ignore',invalid='ignore'):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        if isinstance(precision, np.ndarray):
            precision[(tp + fp)==0] = 0.0
            precision[(tp + fp + fn) == 0] = 1.0
        recall = tp / (tp + fn)
        if isinstance(recall, np.ndarray):
            recall[(tp + fn)==0] = 0.0
            recall[(tp + fp + fn) == 0] = 1.0
        f_one = (2 * precision * recall) / (precision + recall)
        if isinstance(recall, np.ndarray):
            f_one[np.isnan(f_one)] = 0.0
        iou = (tp + SMOOTH) / (tp + fp + fn + SMOOTH)
    confusion_matrices = np.stack((tp, fn, fp, tn), axis=-1)

    micro_metrics_dict = {}
    if isinstance(precision, np.ndarray): # if metrics are np.arrays (i.e. micro metrics from tiles)
            micro_metrics_dict['f_one'] = f_one.ravel() 
            micro_metrics_dict['precision'] = precision.ravel()
            micro_metrics_dict['recall'] = recall.ravel()
            micro_metrics_dict['iou'] = iou.ravel()
            micro_metrics_dict['accuracy'] = accuracy.ravel()
            micro_metrics_dict['tp'], micro_metrics_dict['fn']  = tp.ravel(), fn.ravel()
            micro_metrics_dict['fp'], micro_metrics_dict['tn'] = fp.ravel(), tn.ravel()
    else:
            micro_metrics_dict['f_one'] = f_one 
            micro_metrics_dict['precision'] = precision
            micro_metrics_dict['recall'] = recall
            micro_metrics_dict['iou'] = iou
            micro_metrics_dict['accuracy'] = accuracy
            micro_metrics_dict['tp'], micro_metrics_dict['fn']  = tp, fn
            micro_metrics_dict['fp'], micro_metrics_dict['tn'] = fp, tn

    assert len(np.unique(confusion_matrices.sum(axis=-1))) == 1, "Error! Some predictions seem to have less pixels than the labels!"

    if len(confusion_matrices.shape) == 2: # we are missing the unsqueezed middle dimension
        confusion_matrices = confusion_matrices[:,np.newaxis,:]

    return confusion_matrices, micro_metrics_dict


def compile_micro_metrics(confusion_matrices, print_result=True):
    """
    Collate evaluation metrics by calling corresponding functions. Prints table of metrics.

    :param conf_mat: Confusion matrix produced by conf_mat()
    :return: Dictionary of evaluation metrics.
    """
    SMOOTH = 1e-6
    required_keys = ['tp', 'fn', 'fp', 'tn']
    conf_matrix = dict.fromkeys(required_keys, 0)
    for i, key in enumerate(required_keys):
        try:
            conf_matrix[key] = confusion_matrices[:,:,i].sum() # np.array of micro confusion metrices
        except IndexError:
            conf_matrix[key] = confusion_matrices[i] # single (macro) confusion metrics
    table_entries = np.array([
        ["Truth: slum", conf_matrix['tp'], conf_matrix['fn']],
        ["Truth: non-slum", conf_matrix['fp'], conf_matrix['tn']]
        ])
    headers = ["Confusion matrix", "Prediction: slum", "Prediction: non-slum"]
    if print_result:
        print(tabulate(table_entries, headers, tablefmt="rst", numalign="center"))
    conf_matrix_printable = tabulate(table_entries, headers, tablefmt="rst", numalign="center")

    macro_accuracy =  (conf_matrix['tp'] + conf_matrix['tn']) / (conf_matrix['tp'] + conf_matrix['fp'] + 
                                                                 conf_matrix['fn'] + conf_matrix['tn'])
    macro_precision = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'])
    macro_recall = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'])
    macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
    macro_iou = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'] + conf_matrix['fn'] + SMOOTH)
    
    metrics = {
        "Pixel Accuracy": macro_accuracy,
        "Precision": macro_precision if macro_precision else 0.0,
        "Recall": macro_recall if macro_recall else 0.0,
        "F1 Score": macro_f1 if macro_f1 else 0.0,
        "Intersection over Union": macro_iou}

    metrics_list = list(metrics.items())
    headers = ["Metric", "Value"]
    if print_result:
        print(tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f"))
    metrics_printable = tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f")

    return conf_matrix, conf_matrix_printable, metrics, metrics_printable

def filter_metrics_per_quantiles(precision, recall, f_one, iou, threshold=0.1):
    '''Receive numpy arrays of [precision, recal, f1, iou] and produce a dictionary
       with the 20 "quantile" values (i.e. every 0.05%) for each parameter, and 
       a dictionary with the indices of bottom threshold % [default: 0.05 = 5%] based on each parameter
       The user can use the contents of the tile_indices dictionary to slice the tile_names array and select
       the tiles with the lowest performance.
       Args:
            precision:          np.array, with the micro (i.e. per tile) performance metric
            recall:             np.array, with the micro (i.e. per tile) performance metric
            f_one:              np.array, with the micro (i.e. per tile) performance metric
            iou:                np.array, with the micro (i.e. per tile) performance metric
            threshold:          float[<1.0], the threshold to use for selecting the lowest performing quantile [default: 0.05=5%]
       
       Returns:
            tile_indices:       dict[np.array], 2D index [batch_num, tile_in_batch] of the tiles in the selected lowest quantile
            quantile_results:   dict[np.array], metric value per quantile 
       '''
    quantiles = np.linspace(0,1,num=21)
    tile_indices = {} #defaultdict(np.array)
    quantile_results = {} #defaultdict(np.array)
    for name, values in zip(['lowest_precision', 'lowest_recall', 'lowest_f_one', 'lowest_iou'],[precision, recall, f_one, iou]):
        quantile_results[name] = np.quantile(values, quantiles, interpolation='higher')
        threshold_idx = quantile_results[name][np.searchsorted(quantiles, threshold, side='left').squeeze()]
        tile_indices[name] = np.argwhere(values<=threshold_idx).squeeze()
        ## TODO: check we the dimensions are wrong ?
    quantile_results['quantiles'] = quantiles
    return tile_indices, quantile_results

def ecdf(x):
    '''Returns the % of data points below a certain value:
    Usage:
        >>> cdf = ecdf(data)
        # % of points with value smaller than 0.2
        >>> density_at_0.2 = cdf(0.2)
        '''
    x = np.sort(x)
    def result(v):
        return np.searchsorted(x, v, side='right') / x.size
    return result

def save_tile_overlays(image_name_list, pred_path, dataset_csv, output_path=None):
    '''Overlay predicted and true slum on a tile and save the result
        Args:
            image_name_list:        list[str], a list of images tiles (names only, not full paths)
            pred_path:              str, the path to the folder holding the model predictions   
            dataset_csv:            str, the path to the dataset.csv file used for training/inference
                                    (i.e. tiled_input, and tiled_labels folders)
            output_path:            str, the path to the output folder where results will be saved
                                    (inside the folder 'error_analysis')
        Returns:
            nothing, saves the images to disc'''
    from skimage import io, exposure
    from skimage.draw import rectangle
    from PIL import Image, ImageDraw
    import pandas as pd
    output_path = Path(output_path)
    output_path = output_path if output_path.is_dir() else output_path.parent
    output_path = output_path/'error_analysis'
    pred_path = Path(pred_path)
    pred_path = pred_path if pred_path.is_dir() else pred_path.parent
    os.makedirs(output_path, exist_ok=True)
    dataset_csv = Path(dataset_csv)
    df = pd.read_csv(dataset_csv)
    try:
        for name in image_name_list:
            if (not name.endswith('png')) and (not name.endswith('jpg')):
                continue
            img_path = df.x_location[df.x_location.str.contains(name)].item()
            pred_path_i = str(pred_path/name)
            true_path = df.y_location[df.y_location.str.contains(name)].item()
            img = io.imread(img_path)
            pred = io.imread(pred_path_i)[:,:,0] * 255
            true = io.imread(true_path).squeeze()* 255
            l = 0.6
            img[:,:,0] = ((1-l) * img[:,:,0] + l * pred)
            img[:,:,1] = (1-l) * img[:,:,1] + l * true
            img[:,:,2] = (1-l) * img[:,:,2]
            img = exposure.adjust_gamma(img, l)
            rr, cc = rectangle(start=(0, 5), extent=(30, 40))
            img[rr, cc, 0] = 0
            img[rr, cc, 1] = 0
            img[rr, cc, 2] = 0
            image = Image.fromarray(img)
            imageD = ImageDraw.Draw(image)
            imageD.text((10, 3), 'true', fill=(0,255,0))
            imageD.text((10, 13), 'pred', fill=(255,0,0))
            image.save(output_path/name)
    except Exception as Error:
        print("Error encountered:", Error)
        if Error == 'ValueError':
            print("Warning! Error inspection not possible. Saving tiles with largest errors requires the 'split_tile' flag set to False.\n")


# Prepares images for plotting
def prepare_image(image):
    """
    Prepares tensor input for plotting.

    :param image: Tensor containing image, label or output information.
    :return: Numpy array ready for plotting.
    """
    if not (type(image) is np.ndarray):
        image = image.numpy()

    transposed_image = np.transpose(image, (1,2,0))
    return transposed_image 

# Plots images
def plot_images(images, labels, outputs):
    """
    Plots the images.

    :param images: Tensor containing satellite images.
    :param labels: Tensor containing true image labels.
    :param outputs: Tensor containing model outputs.
    """
    _, axs = plt.subplots(len(images), 3, figsize=(15, 30))

    for i in range(len(images)):
        axs[i, 0].imshow(prepare_image(images[i]))
        axs[i, 0].set_title('Satellite Image')
        
        axs[i, 1].imshow(prepare_image(labels[i]))
        axs[i, 1].set_title('Actual Slum Area')

        axs[i, 2].imshow(prepare_image(outputs[i]))
        axs[i, 2].set_title('Predicted Slum Area')
        
        axs[i, 0].set_xticks([]) # remove ticks and labels on x axis
        axs[i, 0].set_yticks([]) # remove ticks and labels on y axis
        
        axs[i, 1].set_xticks([]) # remove ticks and labels on x axis
        axs[i, 1].set_yticks([]) # remove ticks and labels on y axis

        axs[i, 2].set_xticks([]) # remove ticks and labels on x axis
        axs[i, 2].set_yticks([]) # remove ticks and labels on y axis

    plt.show()

def prepare_masks(tile_size=512, mask_percentage=15, n_masks=5):
    '''Create random masks to be used to mask an input image and use it for self-supervised pre-training.
    ARGS:
        tile_size:          int, the shape of the tiles [default: 512]
        mask_percentage:    int, percentage to mask [default: 15]
        n_masks:            int, number of random masks to prepare [default: 5]
            and (optionaly) the mask percentage
    Returns:
        populates the global variable MASK which will be accessed by the dataloader 
    '''
    global MASK
    shape = (tile_size, tile_size)
    total_pixels = np.prod(shape[:2])
    mask_pixels = int(round(total_pixels * mask_percentage / 100))
    MASK = [np.random.choice(np.arange(total_pixels), size=mask_pixels, replace=False) for _ in range(n_masks)]
    MASK = np.vstack(MASK)

def compare_original_reconstructed(original, reconstructed):
    '''compare an original with a reconstructed map for sanity checking tiling
    args:
        original:       str, path to the original image
        reconstructed:  str, path to the reconstructed image (from tiles)
    usage:
    >>> compare_original_reconstructed('/home/minas/slumworld/data/raw/MD_MUL_97_2016_Brenda/inputs/input_y.png', 
                                       '/home/minas/slumworld/data/tiled/MD_MUL_97_2016_Brenda/reconstructedLabels.png'
    '''
    orig = load_and_fix_slum_map(original)
    rec = load_and_fix_slum_map(reconstructed)
    print("Shape of original:", orig.shape)
    print("Min, max values of original:({},{})".format(orig.min(), orig.max()))
    print("Shape of reconstructed:", rec.shape)
    print("Min, max values of reconstructed:({},{})".format(rec.min(), rec.max()))
    print("Are all elements of the two maps the same? Answer is:", np.equal(orig, rec).all())

def load_and_fix_slum_map(map_file):
    map = io.imread(map_file)
    if map.shape[-1] == 3:
        # 3d image
        map = map[:,:,0]
    elif map.shape[-1] == 1:
        # 3d image
        map = np.squeeze(map, axis=-1)
    if np.max(map) == 255:      # reconstructed slum map for visualization => binarize
        if len(np.unique(map)) == 2:
            map = np.true_divide(map, 255).astype('uint8')
        else:
            print(f"Error! Predicted slum map file {map_file} can only be binary or have values of 0 and 255.")
            print("Aborting operation...")
            sys.exit(1)
    if np.max(map) == 127:     # signed distance map => binarize
        map[map<64] = 0
        map[map>=64] = 1
    return map.astype('uint8')

# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Calculates accuracy
def pixel_acc(conf_mat):
    """
    Compute pixel-level prediction accuracy, i.e. the ratio of true positives plus true negatives to number of pixels.

    Answers the question: "Which share of the pixels did the model predict correctly?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Pixel accuracy score, ranging from 0 to 1.
    """
    if isinstance(conf_mat, dict):
        tp, fp, fn, tn = conf_mat['tp'], conf_mat['fp'], conf_mat['fn'], conf_mat['tn']
    elif isinstance(conf_mat, np.ndarray):
        tp, fp, fn, tn = conf_mat.squeeze()[0], conf_mat.squeeze()[1], conf_mat.squeeze()[2], conf_mat.squeeze()[3]
    else:
        print("Error during metric calculation. Data type not in dict/np.array")
        return None
    pixel_acc = (tp + tn) / (tp + tn + fp + fn)
    return pixel_acc

# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Calculates precision
def precision(conf_mat):
    """
    Compute the precision score, i.e. the ratio of true positives to true positives plus false positives.

    Answers the question: "Which share of the pixels predicted by the model as slum was actually slum?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Precision score, ranging from 0 to 1.
    """
    if isinstance(conf_mat, dict):
        tp, fp, fn, tn = conf_mat['tp'], conf_mat['fp'], conf_mat['fn'], conf_mat['tn']
    elif isinstance(conf_mat, np.ndarray):
        tp, fp, fn, tn = conf_mat.squeeze()[0], conf_mat.squeeze()[1], conf_mat.squeeze()[2], conf_mat.squeeze()[3]
    else:
        print("Error during metric calculation. Data type not in dict/np.array")
        return None
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    return precision

# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Calculates recall
def recall(conf_mat):
    """
    Compute the recall score, i.e. the ratio of true positives to true positives and false negatives.

    Answers the question: "Which share of the pixels that are actually slum was identified by the model as such?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: Recall score, ranging from 0 to 1.
    """
    if isinstance(conf_mat, dict):
        tp, fp, fn, tn = conf_mat['tp'], conf_mat['fp'], conf_mat['fn'], conf_mat['tn']
    elif isinstance(conf_mat, np.ndarray):
        tp, fp, fn, tn = conf_mat.squeeze()[0], conf_mat.squeeze()[1], conf_mat.squeeze()[2], conf_mat.squeeze()[3]
    else:
        print("Error during metric calculation. Data type not in dict/np.array")
        return None
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    return recall

# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Calculates F1 score
def f_one(conf_mat):
    """
    Compute harmonic mean of precision and recall.

    Answers the question: "What is the average of precision and recall?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return:F-1 score, ranging from 0 to 1.
    """
    prec = precision(conf_mat)
    rec = recall(conf_mat)
    if prec + rec == 0:
        f_one = 0
    else:
        f_one = (2 * prec * rec) / (prec + rec)
    return f_one

# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Calculaes IOU score
def iou(conf_mat):
    """
    Compute Intersection over Union (IoU) evaluation metric.

    Answers the question: "What share actual and predicted slum pixels was identified correctly?"

    :param conf_mat: Confusion matrix produced by conf_mat().
    :return: IoU score, ranging from 0 to 1.
    """
    if isinstance(conf_mat, dict):
        tp, fp, fn, tn = conf_mat['tp'], conf_mat['fp'], conf_mat['fn'], conf_mat['tn']
    elif isinstance(conf_mat, np.ndarray):
        tp, fp, fn, tn = conf_mat.squeeze()[0], conf_mat.squeeze()[1], conf_mat.squeeze()[2], conf_mat.squeeze()[3]
    else:
        print("Error during metric calculation. Data type not in dict/np.array")
        return None
    if tp + fp + fn == 0:
        iou = 0
    else:
        iou = tp / (tp + fp + fn)
    return iou

# Based on Tim's code from https://github.com/tim-oh/slum_detector
def compile_metrics(conf_mat, print_result=True):
    """
    Collate evaluation metrics by calling corresponding functions. Prints table of metrics.

    :param conf_mat: Confusion matrix produced by conf_mat()
    :return: Dictionary of evaluation metrics.
    """
    conf_mat['tp'] = float(conf_mat['tp'])
    conf_mat['tn'] = float(conf_mat['tn'])
    conf_mat['fp'] = float(conf_mat['fp'])
    conf_mat['fn'] = float(conf_mat['fn'])
    metrics = {
        "Pixel Accuracy": pixel_acc(conf_mat),
        "Precision": precision(conf_mat),
        "Recall": recall(conf_mat),
        "F1 Score": f_one(conf_mat),
        "Intersection over Union": iou(conf_mat)}
    metrics_list = list(metrics.items())
    headers = ["Metric", "Value"]
    if print_result:
        print(tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f"))
    return metrics, tabulate(metrics_list, headers, tablefmt="rst", numalign="center", floatfmt=".4f")


# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Creates confusion map that is used as base for the confusion matrix
def conf_map(pred, truth):
    
    if not pred.shape == truth.shape:
        raise ValueError("Array sizes: shape of predictions {} must equal shape of ground truth {}.".format(pred.shape, truth.shape))
        
    conf_map = np.empty(pred.shape).astype('str')

    pred = convert_outputs(pred)
    truth = convert_labels(truth)

    if np.any(np.logical_and(pred != 0, pred != 1)):
        raise ValueError("Prediction values: pixels must be 0, 1 or masked")
    if np.any(np.logical_and(truth != 0, truth != 1)):
        raise ValueError("Ground truth values: pixels must be 0, 1 or masked")
    
    tp_index = np.logical_and(pred == 1, truth == 1)
    tn_index = np.logical_and(pred == 0, truth == 0)
    fp_index = np.logical_and(pred == 1, truth == 0)
    fn_index = np.logical_and(pred == 0, truth == 1)

    conf_map[tp_index] = 'tp'
    conf_map[tn_index] = 'tn'
    conf_map[fp_index] = 'fp'
    conf_map[fn_index] = 'fn'

    return conf_map

# Based on Tim's code from https://github.com/tim-oh/slum_detector
# Creates the confusion matrix from a given confusion map
def conf_matrix(conf_map):
    """
    Count sum of pixel-level true positives/false positives/true negatives/false negatives and print results table.

    :param conf_map: Confusion map produced by conf_map().
    :return: Standard confusion matrix, also printed to stdout as a table.
    """
    markers, counts = np.unique(conf_map.data, return_counts=True)
    conf_matrix = dict(zip(markers, counts))

    required_keys = ['fn', 'fp', 'tn', 'tp']
    for key in required_keys:
        try:
            conf_matrix[key]
        except KeyError:
            warnings.warn("Confusion matrix: no %r." % key, UserWarning)
            conf_matrix[key] = 0

    table_entries = np.array([
        ["Truth: slum", conf_matrix['tp'], conf_matrix['fn']],
        ["Truth: non-slum", conf_matrix['fp'], conf_matrix['tn']]
        ])

    headers = ["Confusion matrix", "Prediction: slum", "Prediction: non-slum"]
    print(tabulate(table_entries, headers, tablefmt="rst", numalign="center"))

    return conf_matrix

####################################################################################
###  Additional optimizers from PyTorch_Optimizers:
###  https://pytorch-optimizer.readthedocs.io/en/latest/index.html
###  AdaBound from: https://github.com/Luolc/AdaBound
####################################################################################


Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
Betas2 = Tuple[float, float]
State = Dict[str, Any]
OptFloat = Optional[float]
Nus2 = Tuple[float, float]

class Yogi(Optimizer):
    r"""Implements Yogi Optimizer Algorithm.
    It has been proposed in `Adaptive methods for Nonconvex Optimization`__.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-2)
        betas: coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        initial_accumulator: initial values for first and
            second moments (default: 1e-6)
        weight_decay: weight decay (L2 penalty) (default: 0)

    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.Yogi(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization  # noqa

    Note:
        Reference code: https://github.com/4rtemi5/Yogi-Optimizer_Keras
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-3,
        initial_accumulator: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            initial_accumulator=initial_accumulator,
            weight_decay=weight_decay,
        )
        super(Yogi, self).__init__(params, defaults)

    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Yogi does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )

                state = self.state[p]

                # State initialization
                # Followed from official implementation in tensorflow addons:
                # https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/yogi.py#L118 # noqa
                # For more details refer to the discussion:
                # https://github.com/jettify/pytorch-optimizer/issues/77
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ),
                        group['initial_accumulator'],
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = nn.init.constant_(
                        torch.empty_like(
                            p.data, memory_format=torch.preserve_format
                        ),
                        group['initial_accumulator'],
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_squared = grad.mul(grad)

                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_squared),
                    grad_squared,
                    value=-(1 - beta2),
                )

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps']
                )
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss

class AdaBoundW(Optimizer):
    """Implements AdaBound algorithm with Decoupled Weight Decay (arxiv.org/abs/1711.05101)
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBoundW, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBoundW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    p.data.add_(-step_size)
                    p.data.sub_(decayed_weights)
                else:
                    p.data.add_(-step_size)

        return loss