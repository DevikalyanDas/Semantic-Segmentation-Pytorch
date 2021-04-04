# Semantic Segmentation using PyTorch
## Idea
Semantic Segmentation is about classification of image at the pixel level or in simpler terms, eachpixel is assigned a semantic label. 
## Goal
* To perform semantic segmentation on cityscapes dataset using various deep learning architectures and do a quantitative and qualitative analysis of the performance of the various models.
* Plotting the results.

## Requirements
Every scripts ran in docker container. PyTorch and Torchvision needs to be installed before running the scripts, together with opencv for data-preprocessing and tqdm for showing the training progress. Codes are ran with PyTorch v1.8;

```pip install -r requirements.txt```  

* if you are using docker

  * ``` docker build -f Dockerfile --tag multi_seg .```

  * ``` docker run --gpus all -it -d --rm \```
   ```-v /disk_path/datasets:/app/datasets \```
   ```-v /disk_path/logs/logfiles:/app/logfiles \```
   ```-v /disk_path/logs/checkpoints:/app/checkpoints \```
   ```-v /disk_path/logs/tb-logs:/app/tb-logs \```
   ```multi_seg:latest```


## Approach
I have performed the task of semantic segmentation on images from the CityScapes dataset. To go about this task, U-Net, R2U-Net and Dilated U-Net architectures (More will be added) to compare performances. The metrics I have used is Accuracy(Acc), Sensitivity(SE), Specificity(SP), Dice coeffcient(DSC) and IOU (intersection over union).  
Most of the classes in Cityscapes Dataset out of 30 are void and only 19 classes are available. So, 19 classes were used for training and inference. For training and inference purpose, the dataset was resized to 256 X 256 with a batch size of 8 for models was considered. The dataset has 2975 training and 500 validation images. The validation dataset was used for testing purpose. For, the implementation Pytorch framework was used on multiple GPU clusters (8 Nvidia Tesla V100 GPUs) depending on availability. The models were trained for 200 epochs and the weights were saved after every epochs. The evaluation was performed after each epoch. For the training, Cross-Entropy loss and ADAM optimizer was considered.The performance of Loss for various models during training and validation are illustrated in Figures  
The tranformations which are applied to the training images are random horizontal flip, random rotate (step of 90 deg.) and normalization. The test and validation images are only normalized.
The reported scores are computed considering micro levels.
## Models Used

| R2U-Net     | Unet        | DDU-Net (self-designed)    |
| ----------- | ----------- | -----------| 
| ![runet](https://user-images.githubusercontent.com/14145901/113475253-b644a000-9474-11eb-96b3-335e391f6fa0.png)     |![u-net-architecture](https://user-images.githubusercontent.com/14145901/113475269-ce1c2400-9474-11eb-875c-16d65b88dfd3.png)     |![task3](https://user-images.githubusercontent.com/14145901/113475275-da07e600-9474-11eb-9a64-6f0411008737.png)  |

## How to use
* for r2-unet
  in ```main_r2unet.py```
    * Set the path for the dataset, logs save and checkpoints
    * Also, you can adjust epochs, change optimizers.
    * For training, run ```python3 main_r2unet.py``` and the training will start.
    * For inferencing and visualizing the masks, use ```Inference_output_r2unet.ipynb```
* for U-Net
  in ```main_unet.py```
    * Set the path for the dataset, logs save and checkpoints
    * Also, you can adjust epochs, change optimizers.
    * For training, run ```python3 main_unet.py``` and the training will start.
    * For inferencing and visualizing the masks, use ```Inference_output_unet.ipynb```
* for DDU-Net
  in ```main_ddunet.py```
    * Set the path for the dataset, logs save and checkpoints
    * Also, you can adjust epochs, change optimizers.
    * For training, run ```python3 main_r2unet.py``` and the training will start.
    * For inferencing and visualizing the masks, use ```Inference_output_dduunet.ipynb```    

## Metrice Scores
| Metrices    |    Unet     |    R2U-Net  |   DDU-Net   |
| ----------- | ----------- | ----------- | ----------- | 
|Accuracy     |  0.9899     |   0.9871    | 0.9885      |
|Dice         |  0.9017     |   0.871     |   0.8878    |
|IOU          |  0.8215     |   0.7722    |   0.7985    |
|Sensitivity  |  0.8797     |   0.8296    |    0.8637   |
|Specificity  |    0.996    |   0.9958    |  0.9954     |

## Parameters

| Models      |   Parameters (in Million) |   
| ----------- | ------------------------- | 
|U-Net        | 31.39                     |
|R2U-Net      | 80.94                     |
|DDU-Net      | 10.81                     |


## Various Plots
 * Loss Plots
 ![plot_loss](https://user-images.githubusercontent.com/14145901/113505084-f66f5580-953c-11eb-89a2-27e1f01b681b.png)

 * Accuracy
 ![plot_accuracy](https://user-images.githubusercontent.com/14145901/113505097-16067e00-953d-11eb-9aac-7afc611b7177.png)
 * Dice
 ![plot_dice](https://user-images.githubusercontent.com/14145901/113505106-1dc62280-953d-11eb-90d7-890b3122bdeb.png)
 * IOU
 ![plot_iou](https://user-images.githubusercontent.com/14145901/113505114-23bc0380-953d-11eb-85f8-a0fff85d9914.png)
 * Sensitivity
 ![plot_sens](https://user-images.githubusercontent.com/14145901/113505134-41896880-953d-11eb-96d7-b919116a051d.png)
 * Specificity 
 ![plot_spec](https://user-images.githubusercontent.com/14145901/113505118-29b1e480-953d-11eb-9ef6-208815a2c522.png)

## Segmented Output
| Input      |   Label | U-Net | R2U-Net  | DDU-Net | 
|------------|---------|-------|----------|---------| 
| <img src="sample_imgs/img1/img1.png" width="200" height="200">   | <img src="sample_imgs/img1/mask1.png" width="200" height="200"> | 0| 0  |         0          |
|   0   | 0 | 0| 0  |        0           |
|   0   | 0 | 0| 0  |         0          |
|   0   | 0 | 0| 0  |        0           |
