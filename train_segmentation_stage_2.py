from pickle import FALSE
import torch
from loss import FocalLoss
import os
from torchvision import transforms
import torch.nn.functional as F
import random
from tqdm import tqdm

from dataloader import MVTecDataset, MVTecDataset_cross_validation
from evaluation import evaluation_DRAEM_half

from model_segmentation import DiscriminativeSubNetwork
from model_reconstruction import UNet

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

def add_Gaussian_noise(x, noise_res, noise_std, img_size):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    ns = F.upsample_bilinear(ns, size=[img_size, img_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(128))
    roll_y = random.choice(range(128))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns
    
    return res

def train_on_device(args):
    # run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_colorRange'+'_'+str(args.colorRange)+'_threshold'+'_'+str(args.threshold)
    run_name = args.experiment_name + '_' + args.dataset_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_colorRange'+'_'+str(args.colorRange)+'_threshold'+'_'+str(args.threshold)
    dataset_path = './data/{}'.format(args.dataset_name)
    test_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
    
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

    device = torch.device('cuda:{}'.format(args.gpu_id))
    n_input = 1
    n_classes = 1           # the target is the reconstructed image
    depth = 4
    wf = 6

    model_denoise = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)
    model_segment = DiscriminativeSubNetwork(in_channels=2, out_channels=2).to(device)
    
    model_denoise.to(device)
    model_segment.to(device)
    model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[0])
    model_segment = torch.nn.DataParallel(model_segment, device_ids=[0])
    
    # model_denoise = torch.nn.DataParallel(model_denoise, device_ids=[0, 1])
    # model_segment = torch.nn.DataParallel(model_segment, device_ids=[0, 1])
    
    output_path = './output'
    experiment_path = os.path.join(output_path, run_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
    # ckp_stage_1_path = os.path.join(experiment_path, 'stage_1.pth')
    if args.evaluation:
        ckp_stage_1_path = os.path.join(experiment_path, 'pretrain_stage_1.pth')
    else:
        ckp_stage_1_path = os.path.join(experiment_path, 'stage_1.pth')
    ckp_stage_2_path = os.path.join(experiment_path, 'stage_2.pth')
    result_path = os.path.join(experiment_path, 'results.txt')
    
    # load the pretrained recontrution model from stage 1
    model_denoise.load_state_dict(torch.load(ckp_stage_1_path)['model'])
        
    last_epoch = 0
    if args.resume_training:
        # model_denoise.load_state_dict(torch.load(ckp_stage_1_path)['model_denoise'])
        model_segment.load_state_dict(torch.load(ckp_stage_2_path)['model'])
        last_epoch = torch.load(ckp_stage_2_path)['epoch']
        
    train_data = MVTecDataset(root=dataset_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    test_data = MVTecDataset_cross_validation(root=dataset_path, transform = test_transform, gt_transform=gt_transform, phase='test', data_source=args.experiment_name, args = args)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
        
    loss_l1 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model_segment.parameters(), lr=args.lr)
    loss_focal = FocalLoss()

    for epoch in range(last_epoch, args.epochs):
        model_segment.train()
        loss_list = []
        
        for img, aug, anomaly_mask in tqdm(train_dataloader):
            img = torch.reshape(img, (-1, 1, args.img_size, args.img_size))
            aug = torch.reshape(aug, (-1, 1, args.img_size, args.img_size))
            anomaly_mask = torch.reshape(anomaly_mask, (-1, 1, args.img_size, args.img_size))
            
            img = img.to(device)
            aug = aug.to(device)
            anomaly_mask = anomaly_mask.to(device)

            # Making sure the reconstruction model is not involved
            with torch.no_grad():
                rec = model_denoise(aug)
                rec = rec.detach().cpu()
            rec = rec.to(device)
            joined_in = torch.cat((rec, aug), dim=1)
            
            out_mask = model_segment(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            segment_loss = loss_focal(out_mask_sm, anomaly_mask)
            loss = segment_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f} \n'.format(args.epochs, epoch, mean(loss_list)))
        with open(result_path, 'a') as f:
            f.writelines('epoch [{}/{}], loss:{:.4f} \n'.format(args.epochs, epoch, mean(loss_list)))

        if (epoch) % 5 == 0:
            model_segment.eval()
            auroc_sp, dice_value = evaluation_DRAEM_half(model_denoise, model_segment, test_dataloader, epoch, output_path, run_name, device)
            result_path = os.path.join(output_path, run_name, 'results.txt')
            print('Sample Auroc{:.3f}, Dice{:.3f}'.format(auroc_sp, dice_value))
            with open(result_path, 'a') as f:
                f.writelines('Epoch:{}, Sample Auroc{:.3f}, Dice{:.3f} \n'.format(epoch, auroc_sp, dice_value)) 
            
            # torch.save(model_segment.state_dict(), ckp_path.replace('last', 'segment'))
            torch.save({'model': model_segment.state_dict(),
                        'epoch': epoch}, ckp_stage_2_path)                
            
        

if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=1, action='store', type=int)

    
    # need to be changed/checked every time
    parser.add_argument("-img_size", "--img_size", type=float, default=256,  help="noise magnitude.")
    parser.add_argument('--bs', default = 8, action='store', type=int)
    parser.add_argument('--gpu_id', default='0', action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='Liver_segmentation', action='store')
    parser.add_argument('--colorRange', default=100, action='store')
    parser.add_argument('--threshold', default=200, action='store')
    parser.add_argument('--dataset_name', default='liver_dataset', action='store')
    parser.add_argument('--rejection', default=False, action='store')
    parser.add_argument('--number_iterations', default=1, action='store')
    parser.add_argument('--control_texture', default=False, action='store')
    parser.add_argument('--cutout', default=False, action='store')
    parser.add_argument('--resume_training', default=False, action='store')
    parser.add_argument('--evaluation', default=False, action='store')
    args = parser.parse_args()
   
    # Use if need dual gpu training
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # if args.gpu_id is None:
    #     gpus = "0"
    #     os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    # else:
    #     gpus = ""
    #     for i in range(len(args.gpu_id)):
    #         gpus = gpus + args.gpu_id[i] + ","
    #     os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]
    # torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    # with torch.cuda.device(args.gpu_id):
    train_on_device(args)
