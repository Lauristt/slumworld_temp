## Model Architecture

All models in this project share the U-Net architecture. The original paper can be found [here](https://arxiv.org/pdf/1505.04597.pdf). The U-Net architecture is ideal for image segmentation tasks when the number of available data is small like in our case. As stated in the paper, a U-Net *"consists of a contracting path to capture context and a symmetric expanding path that enables precise localization."*
 
Apart from the vanilla model implementation, in order to get better results, we used pretrained networks for the encoding (contracting) part of the U-Net. In particular,
we used pretrained VGG models (VGG11, VGG13, VGG16, VGG19) and pretrained ResNet models (ResNet18, ResNet34, ResNet50).
 
VGG stands for Visual Geometry Group. The original paper for the VGG model can be found [here](https://arxiv.org/pdf/1409.1556.pdf).
 
ResNet stands for Residual Network. The original paper for the ResNet model can be found [here](https://arxiv.org/pdf/1512.03385.pdf).
ResNets are deeper than VGGs but have lower complexity, thus they achieve lower training times.
 
Finally we used DeepLabV3 model, the original paper for which can be found [here](https://arxiv.org/pdf/1706.05587.pdf).

