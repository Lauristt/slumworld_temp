import copy
import numpy as np
from torchvision import transforms
try:
    import slumworldML.src.custom_transformations as ct
except Exception as Error:
    try:
        import custom_transformations as ct
    except Exception as Error:
        from src import custom_transformations as ct
    # from cnn_tiler import CNNTiler


###############################################################################################
# Once model is finalised hard code mean and std here as python list
MEAN = np.array([0.])
STD = np.array([1.])

# MEAN = [0, 0, 0] 
# STD = [1, 1, 1]

################################################################################################
# mean and std need to be passed at run time during traning. Do not add normalisation transform. This is added in
# create_transform function.


def create_transform(transformation, mean=MEAN, std=STD, input_size=256):
    """Creates a callable transformation class. x_transform SHOULD NOT include normalisation. Mean and std need
    to be passed after this module is imported during tuning/traning phase."""
    transformation_ = copy.deepcopy(transformation)
    norm = transforms.Normalize(mean=mean, std=std)
    if len(transformation_['x_transforms']) == 0:
        transformation_['x_transforms'].append( ct.ZeroToOneRange())
    transformation_['x_transforms'].append(norm)

    return ct.SegmentationCompose(transformation_['x_transforms'], transformation_['joint_transforms'], 
                                  input_tile_size=input_size, normalization_stats=(mean, std))


TRAINING_TRANSFORMS_BASIC = {
    "x_transforms" : [ct.ZeroToOneRange()],
    "joint_transforms" : [  ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize()]
}

TRAINING_TRANSFORMS = {
    "x_transforms" : [transforms.RandomEqualize(p=0.33),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.RandomInvert(p=0.33),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.05, 3)),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=0.85, p=0.33),
                      transforms.RandomGrayscale(p=0.33)
                      ],
    "joint_transforms" : [  ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize()]
}

SSP_TRANSFORMS = {
    "x_transforms" : [transforms.RandomEqualize(p=0.33),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.RandomInvert(p=0.33),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.05, 3)),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=192.0, p=0.33),
                      ],
    "joint_transforms" : [  ct.SSP_Generator(),
                            ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize(),]
}
ASSP_TRANSFORMS = {
    "x_transforms" : [transforms.RandomEqualize(p=0.33),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.RandomInvert(p=0.33),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.05, 3)),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=192.0, p=0.33),
                      ],
    "joint_transforms" : [  ct.Autoencoding_SSP_Generator(),
                            ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize(),]
}

SSP_TRANSFORMS_PAN = {
    "x_transforms" : [transforms.RandomEqualize(p=0.33),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=192.0, p=0.33)
                    ],
    "joint_transforms" : [  ct.SSP_Generator(),
                            ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize(),]
}
ASSP_TRANSFORMS_PAN = {
    "x_transforms" : [transforms.RandomEqualize(p=0.33),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=192.0, p=0.33)
                    ],
    "joint_transforms" : [  ct.Autoencoding_SSP_Generator(),
                            ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize(),]
}

### These are the active ones
TRAINING_TRANSFORMS = {
    "x_transforms" : [
                      transforms.RandomEqualize(p=0.2),
                      ct.ZeroToOneRange(),
                      ct.ProbWrapper(prob=0.75, transformation=transforms.ColorJitter(brightness=[0.3,1.5], 
                                                                                     contrast=[0.4,2], 
                                                                                     saturation=[0.,0.5], 
                                                                                     hue=[-0.05,0.05])),

                      transforms.RandomAdjustSharpness(2,p=0.5),
                      ct.RandomChannelShuffle(shuffle_probability=.1),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.05, 1)),
                      transforms.RandomGrayscale(p=0.1),
                      ],
    "joint_transforms" : [  
                        # ct.LabelNoiseFromDistances(),
                        ct.RandomHflip(),
                        ct.RandomVflip(),
                        ct.RandomRotateZoomCropTensor(),
                        ct.Resize()
                         ]}

TRAINING_TRANSFORMS_PAN = {
    "x_transforms" : [transforms.RandomEqualize(p=0.5),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 3)),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=0.85, p=0.2)
                    ],
    "joint_transforms" : [  
                            # ct.LabelNoiseFromDistances(),
                            ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize()]
}
# coppied PAN transforms for PANS
TRAINING_TRANSFORMS = {
    "x_transforms" : [transforms.RandomEqualize(p=0.5),
                      ct.ZeroToOneRange(),
                      transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
                      transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 3)),
                      ct.RandomChannelShuffle(shuffle_probability=.1),
                      transforms.RandomAutocontrast(p=0.33),
                      transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                      transforms.RandomSolarize(threshold=0.85, p=0.2)
                    ],
    "joint_transforms" : [  
                            # ct.LabelNoiseFromDistances(),
                            ct.RandomHflip(),
                            ct.RandomVflip(),
                            ct.RandomRotateZoomCropTensor(),
                            ct.Resize()]
}
### TODO:
###      TEST TrivialAugmentBasic

# TRAINING_TRANSFORMS = {
#     "x_transforms" : [
#                       ],
#     "joint_transforms" : [  
#                         ct.TrivialAugment(fill=None),
#                         ct.ZeroToOneRange(),                        
#                         ct.LabelNoise(0.1),
#                         ct.RandomRotateZoomCropTensor(),
#                         ct.Resize()
#                          ]}

# TRAINING_TRANSFORMS_PAN = {
#     "x_transforms" : [ct.ZeroToOneRange(),
#                       transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
#                       transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3)),
#                     ],
#     "joint_transforms" : [  ct.RandomHflip(),
#                             ct.RandomVflip(),
#                             ct.RandomRotateZoomCropTensor(),
#                             ct.Resize()]
# }
TTA_MUL_TRANSFORMS_OLD = {
    "x_transforms" : [ct.ZeroToOneRange(),
            transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.05, 1)),
            transforms.RandomInvert(p=0.33),
            transforms.RandomGrayscale(p=0.1),
    ],
    "joint_transforms" : []
}
INFERENCE_TRANSFORMS = {
    "x_transforms" : [ct.ZeroToOneRange()],
    "joint_transforms" : [] #[ct.BinarizeLabels()]
}

TTA_MUL_TRANSFORMS = {
    "x_transforms" : TRAINING_TRANSFORMS["x_transforms"],
    "joint_transforms" : [] #[ct.BinarizeLabels()]
}
# TTA_MUL_TRANSFORMS = TTA_MUL_TRANSFORMS_OLD
TTA_PAN_TRANSFORMS = {
    "x_transforms" : TRAINING_TRANSFORMS_PAN["x_transforms"],
    "joint_transforms" : []
}
SSP_INFERENCE_TRANSFORMS = {
    "x_transforms" : [ct.ZeroToOneRange()],
    "joint_transforms" : [ct.SSP_Generator()]
}

ASSP_INFERENCE_TRANSFORMS = {
    "x_transforms" : [ct.ZeroToOneRange()],
    "joint_transforms" : [ct.Autoencoding_SSP_Generator()]
}

VALIDATION_TRANSFORM =  {
    "x_transforms" : [ct.ZeroToOneRange()],
    "joint_transforms" : []
}




INFERENCE_TRANSFORMS_DICT = {'inference_mul': INFERENCE_TRANSFORMS,
                             'inference_pan': INFERENCE_TRANSFORMS,
                             'tta_mul': TTA_MUL_TRANSFORMS,
                             'tta_pan': TTA_PAN_TRANSFORMS,
                             'inference_ssp': SSP_INFERENCE_TRANSFORMS,
                             'inference_assp':ASSP_INFERENCE_TRANSFORMS}
