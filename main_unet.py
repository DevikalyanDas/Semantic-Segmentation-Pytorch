import os
import numpy as np
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

from statistics import mean
import time
from tqdm import tqdm
import warnings

from vis_task_3_transforms import Compose, RandomHorizontalFlip, RandomRotate, ToTensor,Normalize
from vis_task_3_utils import recursive_glob,create_data_folder
from data_city import cityscapesLoader
from model_unet import real_unet, init_weights
from metrices1 import IoU,Fscore,Accuracy,Sensitivity,Specificity

#####folder paths for dataset, logs, checkpoints (change a.p.y requirement) #####
local_path = '/app/datasets/data_folder/'   # for accessing data
ckp_path = '/app/checkpoints/cv_unet/'     # checkpoint save path 
logs_path = '/app/logfiles/cv_unet_logs'   # logs save path
###### data folder create #####

dst_train = cityscapesLoader(local_path, is_transform=True,split="train",img_size=(256,256))
bs = 8  # batch size
trainloader = data.DataLoader(dst_train, batch_size=bs, num_workers=0,shuffle=True)

dst_val = cityscapesLoader(local_path, is_transform=False,split="val",img_size=(256,256))
val_loader = data.DataLoader(dst_val, batch_size=bs, num_workers=0,shuffle=True)

############ Model create and Initialization #############
# calling the Dilated Unet model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = real_unet(n_class=19)
model.apply(init_weights)
#model = torch.nn.DataParallel(model)
model = model.to(device)

#summary(model, (3, 256, 256))


############loss and optimizer###################
# loss function
loss_f = torch.nn.CrossEntropyLoss()

# optimizer variable
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# parameters 
epochs = 200

############ checkpoints save and evaluation metrices csv save path

warnings.filterwarnings("ignore")
def main():

    loss_history={"epoch":[],"train": [],"val":[]}
    train_metrices_history={"epoch":[],"acc": [],"iou":[],"dice":[],"sens":[],"spec":[]}
    val_metrices_history={"epoch":[],"acc": [],"iou":[],"dice":[],"sens":[],"spec":[]}
    prev_loss = -100
    loss_increase_counter = 0
    early_stop = True
    early_stop_threshold = 5
    
    for e in range(epochs):
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0
        train_sensitivity = 0.0
        train_specificity = 0.0
        train_iou = 0.0 # Jaccard Score
        train_dice = 0.0
        ts = time.time()

        for i, d in tqdm(enumerate(trainloader),total=len(trainloader),leave=True,position=0,desc='Epoch: {}'.format(e)):
            inputs_, targets_,_ = d
            inputs, targets = inputs_.to(device), targets_.to(device)

            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_f(outputs, targets.long())
            loss.backward()     
            opt.step()
            running_loss += loss.item()
            
            train_iou += IoU()(outputs,targets)
            train_dice += Fscore()(outputs,targets)
            train_accuracy += Accuracy()(outputs,targets)
            train_sensitivity += Sensitivity()(outputs,targets)
            train_specificity += Specificity()(outputs,targets)
            
        state = {'epoch': e,'state_dict': model.state_dict(),
                 'optimizer':  opt.state_dict(),'loss': loss.item()}
        
        tr_loss = running_loss/len(trainloader)
        
        vl_loss, val_acc,val_iou,val_dice,val_sens,val_spec = val()    
        
        loss_history["epoch"].append(e)
        loss_history["train"].append(tr_loss)
        loss_history["val"].append(vl_loss)

        
        tr_acc = train_accuracy/len(trainloader)
        tr_iou = train_iou/len(trainloader)
        tr_dice = train_dice/len(trainloader)
        tr_sens = train_sensitivity/len(trainloader)
        tr_spec = train_specificity/len(trainloader)

        val_metrices_history["epoch"].append(e)
        val_metrices_history["acc"].append(val_acc)
        val_metrices_history["iou"].append(val_iou)
        val_metrices_history["dice"].append(val_dice)
        val_metrices_history["sens"].append(val_sens)
        val_metrices_history["spec"].append(val_spec)

        train_metrices_history["epoch"].append(e)
        train_metrices_history["acc"].append(tr_acc)
        train_metrices_history["iou"].append(tr_iou)
        train_metrices_history["dice"].append(tr_dice)
        train_metrices_history["sens"].append(tr_sens)
        train_metrices_history["spec"].append(tr_spec)

        file_name = os.path.join(ckp_path+'task_2_weights_epoch_{}'.format(e) + '.pt')

        print("Finish Epoch {0},Time Elapsed {1}, train loss: {2:.6g}, val loss: {3:.6g}".format(e,time.time() - ts,tr_loss,vl_loss))
        print("Metrices Train : Acc {0:.6g},IOU {1:.6g}, Dice {2:.6g}, Sens {3:.6g}, Spec  {4:.6g}".format(tr_acc,tr_iou,tr_dice,tr_sens,tr_spec))
        print("Metrices Valid : Acc {0:.6g},IOU {1:.6g}, Dice {2:.6g},  Sens {3:.6g}, Spec  {4:.6g}".format(val_acc,val_iou,val_dice,val_sens,val_spec))    
        print("-"*60)
        torch.save(state, file_name)
        # Implemented early stopping
        if vl_loss > prev_loss:
            loss_increase_counter += 1
        else:
            loss_increase_counter = 0
        if early_stop and loss_increase_counter > early_stop_threshold:
            print("Early Stopping..")
            break
        
        prev_loss = vl_loss

        torch.cuda.empty_cache()
        
        list(getattr(tqdm, '_instances'))

        for instance in list(tqdm._instances):
            tqdm._decr_instances(instance)

    print('Finished Training')

    # Write the train loss to the csv
    (pd.DataFrame.from_dict(data=loss_history, orient='columns').to_csv(os.path.join(logs_path,'loss.csv'), header=['epoch','train_loss','val_loss']))
    (pd.DataFrame.from_dict(data=train_metrices_history, orient='columns').to_csv(os.path.join(logs_path,'train_metrices.csv'), header=["epoch","acc","iou","dice","sens","spec"]))
    (pd.DataFrame.from_dict(data=val_metrices_history, orient='columns').to_csv(os.path.join(logs_path,'val_metrices.csv'), header=["epoch","acc","iou","dice","sens","spec"]))
def val():
    model.eval()

    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    # Evaluate

    val_loss = 0.0
    
    val_accuracy = 0.0
    val_sensitivity = 0.0
    val_specificity = 0.0
    val_iou = 0.0 # Jaccard Score
    val_dice = 0.0 
    with torch.no_grad():
        for iter, d in tqdm(enumerate(val_loader)):
            inputs_, labels_, _ = d
            inputs, targets = inputs_.to(device), labels_.to(device)

            outputs = model(inputs)

            loss = loss_f(outputs, targets.long())
            val_loss += loss.item()
            
            val_iou += IoU()(outputs,targets)
            val_dice += Fscore()(outputs,targets)
            val_accuracy += Accuracy()(outputs,targets)
            val_sensitivity += Sensitivity()(outputs,targets)
            val_specificity += Specificity()(outputs,targets)
  
    return (val_loss/len(val_loader)),(val_accuracy/len(val_loader)), (val_iou/len(val_loader)), (val_dice/len(val_loader)),(val_sensitivity/len(val_loader)),(val_specificity/len(val_loader))
               
if __name__ == '__main__':
    main()
