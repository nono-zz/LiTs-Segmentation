from pickle import FALSE
import torch
import os
from torchvision import transforms
from tqdm import tqdm

import torch.nn.functional as F
import random

from dataloader import MVTecDataset, MVTecDataset_cross_validation
from evaluation import evaluation_reconstruction

from model_reconstruction import UNet

from torchvision.utils import save_image
    
def mean(list_x):
    return sum(list_x)/len(list_x)
        
def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])

    return data_transforms, gt_transforms

def train_on_device(args):
    run_name = args.experiment_name + '_' + args.dataset_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_colorRange'+'_'+str(args.colorRange)+'_threshold'+'_'+str(args.threshold)
    dataset_path = './data/{}'.format(args.dataset_name)
    test_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
    
    # Retrieve the train/test/label directory.
    dirs = os.listdir(dataset_path)
    for dir_name in dirs:
        if 'train' in dir_name:
            train_dir = dir_name
        elif 'test' in dir_name:
            if 'label' in dir_name:
                label_dir = dir_name
            else:
                test_dir = dir_name
    if 'label_dir' in locals():
        dirs = [train_dir, test_dir, label_dir]                

    # Reconstruction model configuration
    n_input = 1
    n_classes = 1
    depth = 4
    wf = 6
    model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])     # push model to parallel gpu
    
    # Ouput and result saving dir
    output_path = './output'
    experiment_path = os.path.join(output_path, run_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    ckp_path = os.path.join(experiment_path, 'stage_1.pth')
    result_path = os.path.join(experiment_path, 'results.txt')
    
    # Load saved checkpoint if set to true
    last_epoch = 0
    if args.resume_training:
        model.load_state_dict(torch.load(ckp_path)['model'])
        last_epoch = torch.load(ckp_path)['epoch']
    
    # Dataset initialization
    train_data = MVTecDataset(root=dataset_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    test_data = MVTecDataset(root='../dataset/hist_DIY', transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name, args = args)
    # test_data = MVTecDataset_cross_validation(root=dataset_path, transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
    
    # Loss & optimizer
    loss_l1 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    # Start training
    for epoch in range(last_epoch, args.epochs):
        model.train()
        loss_list = []
        
        for img, aug, anomaly_mask in tqdm(train_dataloader):
            img = torch.reshape(img, (-1, 1, args.img_size, args.img_size))
            aug = torch.reshape(aug, (-1, 1, args.img_size, args.img_size))
            anomaly_mask = torch.reshape(anomaly_mask, (-1, 1, args.img_size, args.img_size))
            
            img = img.cuda()
            aug = aug.cuda()
            anomaly_mask = anomaly_mask.cuda()
            rec = model(aug)
            
            loss = loss_l1(rec,img)
            
            
            save_image(aug, 'aug.png')
            save_image(rec, 'rec_output.png')
            save_image(img, 'rec_target.png')
            save_image(anomaly_mask, 'mask_target.png')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(loss_list)))
        with open(result_path, 'a') as f:
            f.writelines('epoch [{}/{}], loss:{:.4f} \n'.format(args.epochs, epoch, mean(loss_list)))   
        
        # Evaluate every 10 epochs: reconstruction model evaluation, if needed.     
        if (epoch) % 10 == 0:
            model.eval()
            dice_value, auroc_sp = evaluation_reconstruction(model, test_dataloader, epoch, run_name, output_path)
            result_path = os.path.join(output_path, run_name, 'results.txt')
            # Print and write results
            print('Sample Auroc{:.3f}, Dice{:3f}'.format(auroc_sp, dice_value))
            with open(result_path, 'a') as f:
                f.writelines('Epoch:{}, Sample Auroc{:.3f}, Dice:{:3f} \n'.format(epoch, auroc_sp, dice_value))   
            # Save model
            torch.save({'model': model.state_dict(),
                        'epoch': epoch}, ckp_path)
        
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=201, action='store', type=int)
    
    # Training configs
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--gpu_id', default=['0','1'], action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='Liver_segmentation', action='store')
    parser.add_argument('--colorRange', default=100, action='store')
    parser.add_argument('--threshold', default=200, action='store')
    parser.add_argument('--dataset_name', default='liver_dataset', action='store')
    parser.add_argument('--model', default='ws_skip_connection', action='store')
    parser.add_argument('--multi_layer', default=False, action='store')
    parser.add_argument('--rejection', default=False, action='store')
    parser.add_argument('--number_iterations', default=1, action='store')
    parser.add_argument('--control_texture', default=False, action='store')
    parser.add_argument('--cutout', default=False, action='store')
    parser.add_argument('--resume_training', default=False, action='store')
    
    args = parser.parse_args()
   
    # Dual gpu parallel training
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpu_id is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpu_id)):
            gpus = gpus + args.gpu_id[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    train_on_device(args)
