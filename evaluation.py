import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm

import os

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def mean(list_x):
    return sum(list_x)/len(list_x)

def cal_distance_map(input, target):
    # input = np.squeeze(input, axis=0)
    # target = np.squeeze(target, axis=0)
    d_map = np.full_like(input, 0)
    d_map = np.square(input - target)
    return d_map

def dice(pred, gt):
    intersection = (pred*gt).sum()
    return (2. * intersection)/(pred.sum() + gt.sum())


def dice_tensor(a,b):
    num = 2 * (a & b).sum()
    den = a.sum() + b.sum()
    den_float = den.float()
    den_float[den == 0] = float("nan")
     
    return num.float() / den_float

def evaluation_DRAEM_half(model_denoise, model_segment, test_dataloader, epoch, output_path, run_name, device, threshold = 0.5):
    
    model_denoise.eval()
    model_segment.eval()
    
    y_true_ = torch.zeros(256*256*len(test_dataloader), dtype=torch.half)
    y_pred_ = torch.zeros(256*256*len(test_dataloader), dtype=torch.half)
    y_sample_true_ = torch.zeros(len(test_dataloader), dtype=torch.half)
    y_sample_pred_ = torch.zeros(len(test_dataloader), dtype=torch.half)
    
    y_sample_pred_mean_ = torch.zeros(len(test_dataloader), dtype=torch.half)
    y_sample_true_mean_ = torch.zeros(len(test_dataloader), dtype=torch.half)
    i = 0
    j = 0
    
    with torch.no_grad():
        for img, gt, label, img_path, save in tqdm(test_dataloader):

            img = img.to(device)
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            
            y_ = gt.view(-1)
            # check if img is RGB
            rec = model_denoise(img)
                    
            joined_in = torch.cat((rec, img), dim=1)
            
            out_mask = model_segment(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            # save_image(img, 'eval_raw.png')
            # save_image(rec, 'eval_rec.png')
            # save_image(out_mask_sm[:,1:,:,:], 'eval_mask_output.png')
            # save_image(gt, 'eval_gt.png')

            anomaly_map = out_mask_sm
            
            gt = gt[0,0,:,:].to('cpu').detach().numpy()  
            anomaly_map = anomaly_map[0,1,:,:].to('cpu').detach().numpy()  
            
            # binarize the anomaly map
            # anomaly_map = np.where(anomaly_map > 0.5, 1, 0)
            y_hat = torch.from_numpy(anomaly_map)
            y_hat = y_hat.reshape(-1)
        
            y_sample_true_[j] = (y_.max()).half()
            y_sample_true_mean_[j] = (y_.max()).half()
            y_sample_pred_[j] = (y_hat.max()).half()
            y_sample_pred_mean_[j] = (y_hat.sum()).half()
            
            y_true_[i:i + y_.numel()] = y_.half()
            y_pred_[i:i + y_hat.numel()] = y_hat.half()
            i += y_.numel()
            j += 1
         
         
        # y_sample_true_ = y_sample_true_.to(device)   
        # y_sample_pred_ = y_sample_pred_.to(device)
        y_true_ = y_true_.to(device)
        y_pred_ = y_pred_.to(device)

        # Use another gpu if gpu memory is full
        # device_dice = torch.device('cuda:{}'.format('0'))
        # y_true_ = y_true_.to(device_dice)
        # y_pred_ = y_pred_.to(device_dice)
            
        dice_value = dice_tensor(y_true_ > 0.5, y_pred_ > 0.5).cpu().item()
        auroc_sp = round(roc_auc_score(y_sample_true_, y_sample_pred_), 3)
        
        del y_true_
        del y_pred_
    
    return auroc_sp, dice_value

def evaluation_reconstruction(model, test_dataloader, epoch, run_name, output_path, threshold = 0.1):
    
    model.eval()
    
    gt_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    pr_binary_list_px = []

    with torch.no_grad():
        for img, gt, label, img_path, save in tqdm(test_dataloader):
            img = img.cuda()
            gt[gt > 0.1] = 1
            gt[gt <= 0.1] = 0
            rec = model(img)
            difference = cal_distance_map(rec[0,0,:,:].to('cpu').detach().numpy(), img[0,0,:,:].to('cpu').detach().numpy())
            
            gt = gt[0,0,:,:].to('cpu').detach().numpy()  
            
            prediction_map = np.where(difference > threshold, 1, 0)
            gt_list_px.extend(gt.astype(int).ravel())
            pr_binary_list_px.extend(prediction_map.ravel())
            gt_list_sp.append(np.max(gt.astype(int)))
            pr_list_sp.append(np.max(difference))
            
        dice_value = dice(np.array(gt_list_px), np.array(pr_binary_list_px))
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
   
    return dice_value, auroc_sp