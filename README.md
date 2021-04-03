# Semantic Segmentation using PyTorch
## Idea
Semantic Segmentation is about classification of image at the pixel level or in simpler terms, eachpixel is assigned a semantic label. 
## Goal
1. To perform semantic segmentation on cityscapes dataset using various deep learning architectures and do a quantitative and qualitative analysis of the performance of the various models.
2. Plotting the results.

## Requirements
Every scripts ran in docker container. PyTorch and Torchvision needs to be installed before running the scripts, together with opencv for data-preprocessing and tqdm for showing the training progress. Codes are ran with PyTorch v1.8;

```pip install -r requirements.txt```
## Approach
I have performed the task of semantic segmentation on images from the CityScapes dataset. To go about this task, U-Net, R2U-Net and Dilated U-Net architectures (More will be added) to compare performances. The metrics I have used is Accuracy(Acc), Sensitivity(SE), Specificity(SP), Dice coeffcient(DSC) and IOU (intersection over union). 



