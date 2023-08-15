from PIL import Image, ImageOps
import os
import torch
import glob
import numpy as np

from anomaly_sythesis import colorJitterRandom


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, dirs, data_source = 'liver', rgb=False, args=None):
        if len(dirs) == 3:
            [train_dir, test_dir, label_dir] = dirs
        elif len(dirs) == 2:
            [train_dir, test_dir] = dirs
            
        self.phase = phase
        self.transform = transform
        self.args = args
        
        self.gt_transform = gt_transform
        self.data_source = data_source
        self.rgb = rgb
        
        if phase == 'train':
            self.img_path = os.path.join(root, 'train/good')
            # self.img_paths = glob.glob(self.img_path + "/*.png")
            if data_source == 'retina':
                self.img_paths = glob.glob(self.img_path + "/*.bmp")
            else:
                self.img_paths = glob.glob(self.img_path + "/*.png")
            self.img_paths.sort()
        elif phase == 'test':
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'test_label')
            # load dataset
            self.img_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
            
        elif phase == 'eval':
            self.img_path = os.path.join(root, 'evaluation')
            self.gt_path = os.path.join(root, 'evaluation_label')
            self.img_paths, self.gt_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):         # only used in test phase
        
        img_tot_paths = []
        gt_tot_paths = []

        img_paths = glob.glob(self.img_path + "/*.png")
        gt_paths = glob.glob(self.gt_path + "/*.png")            # ground truth mask.
        img_paths.sort()
        gt_paths.sort()
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend(gt_paths)

        assert len(img_tot_paths) == len(gt_tot_paths), "Number of test and ground truth pair doesn't match!"

        return img_tot_paths, gt_tot_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        if self.phase == 'train':
            img_path = self.img_paths[idx]
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            
            """ Sample level augmentation"""
            img_numpy = np.array(img)
            # cv2.imwrite('img_numpy.png', img_numpy)
            if img_numpy.max() == 0:
                img_tensor = self.transform(img)
                img_tensor = img_tensor.repeat(2, 1, 1, 1)
                gt_tensor = torch.zeros_like(img_tensor)
                return img_tensor, img_tensor
            
           # Image augumentation
            colorJitter_img, colorJitter_gt = colorJitterRandom(img_numpy, self.args, colorRange=self.args.colorRange, threshold=self.args.threshold, number_iterations=self.args.number_iterations, cutout = self.args.cutout)
            colorJitter_img = np.expand_dims(colorJitter_img, axis=2)
            
            colorJitter_img = self.transform(colorJitter_img)
        
            # Format conversion
            org = self.transform(img) 
            img_list = [colorJitter_img]
            gt_list = [colorJitter_gt]
            gt_list = [(np.where(x > 0, 255, 0)).astype(np.uint8) for x in gt_list]
            gt_list = [self.transform(x) for x in gt_list]
            gt_list = [x.unsqueeze(dim=0) for x in gt_list]
            gt_tensor = torch.cat(gt_list, dim=0)
            
            img_list = [x.unsqueeze(dim=0) for x in img_list]
            aug_tensor = torch.cat(img_list, dim = 0)
            
            org_tensor = (torch.unsqueeze(org, dim=0)).repeat(len(img_list), 1, 1, 1)
            return org_tensor, aug_tensor, gt_tensor
        
        else:
            img_path, gt_path= self.img_paths[idx], self.gt_paths[idx]
            # img = Image.open(img_path).convert('RGB')s
            img = Image.open(img_path)
            img = ImageOps.grayscale(img)
            img = img.resize([self.args.img_size, self.args.img_size])
            if self.rgb:    
                img = img.convert('RGB')
            img = self.transform(img)
            
            gt = Image.open(gt_path)
            gt = ImageOps.grayscale(gt)
            gt = gt.resize([self.args.img_size, self.args.img_size])
            gt = self.gt_transform(gt)

            # determine the label
            if torch.sum(gt) != 0:
                label = 1
            else:
                label = 0
                
            save = False
            return img, gt, label, img_path, save
        
     
"""Designed for testing in cross validation
"""   
class MVTecDataset_cross_validation(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, data_source = 'liver', rgb=False, args=None, fold_index = 0, test_whole=False):
        self.phase = phase
        self.transform = transform
        self.args = args
        self.root = root
        
        self.gt_transform = gt_transform
        self.data_source = data_source
        self.rgb = rgb
        
        file_path = os.path.join(root, 'fold_{}.npy'.format(fold_index))
        data = np.load(file_path, allow_pickle=True)
        
        self.train_names = data.item()['Train_images']
        self.test_names = data.item()['Test_images']
        
        self.train_labels = data.item()['Train_labels']
        self.test_labels = data.item()['Test_labels']
        
        # Retrieve all the healthy images as training samples
        self.label_mask = np.array(self.train_labels)
        self.label_mask = np.where(self.label_mask == 1, False, True)
        self.train_names = self.train_names[self.label_mask]
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.train_names)
        
        else:
            return len(self.test_names)

    def __getitem__(self, idx):
       
        # img_path, gt_path= self.img_paths[idx], self.gt_paths[idx]
        img_name = self.test_names[idx]
        img_path = os.path.join(self.root, 'image', img_name)
        
        gt_name = img_name.replace('liver', 'liver_gt')
        gt_path = os.path.join(self.root, 'image_label', gt_name)
        
        img = Image.open(img_path)
        img = ImageOps.grayscale(img)
        img = img.resize([self.args.img_size, self.args.img_size])
        if self.rgb:    
            img = img.convert('RGB')
        img = self.transform(img)
        
        gt = Image.open(gt_path)
        gt = ImageOps.grayscale(gt)
        gt = gt.resize([self.args.img_size, self.args.img_size])
        gt = self.gt_transform(gt)

        # Determine the label
        if torch.sum(gt) != 0:
            label = 1
        else:
            label = 0
            
        save = False
        return img, gt, label, img_path, save
    
        
