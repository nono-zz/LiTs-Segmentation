import numpy as np
import cv2
import random

from torchvision import transforms
from torchvision.utils import save_image

import skimage.exposure
import numpy as np
from numpy.random import default_rng

"""random shape
"""
def randomShape(img, scaleUpper=255, threshold=200):


    # define random seed to change the pattern
    rng = default_rng()

    # define image size
    width=img.shape[0]
    height=img.shape[1]

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    # thresh = cv2.threshold(stretch, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    mask = img > 0.01
    
    anomalyMask = mask * result
    anomalyMask = np.where(anomalyMask > 0, 1, 0)
    
    addImg = np.ones_like(img)
    scale = random.randint(0,scaleUpper)
    
    augImg = img * (1-anomalyMask) + addImg * anomalyMask * scale
    return augImg.astype(np.uint8), anomalyMask.astype(np.uint8), stretch.astype(np.uint8)


def colorJitterRandom(img, args, colorRange = 100, minscale = 60, colorjitterScale=0, threshold=200, number_iterations=1, control_texture=False, cutout=False, min_area = 80, min_img_area = 200):
    colorJitter_fn = transforms.ColorJitter(brightness = colorjitterScale,
                                                      contrast = colorjitterScale,
                                                      saturation = colorjitterScale,
                                                      hue = colorjitterScale)
    img = cv2.resize(img, [256, 256])
    tot_gt_mask = np.zeros_like(img)
    
    for i in range(number_iterations):
        new_img, gt_mask, randomMap = randomShape(img, threshold=threshold)
        
        
        if args.rejection:
            while gt_mask.sum() == 0 :
                new_img, gt_mask, randomMap = randomShape(img, threshold=threshold)
        
        tot_gt_mask = tot_gt_mask + gt_mask
        
    tot_gt_mask = np.where(tot_gt_mask > 0, 1, 0)
    
    while abs(colorjitterScale) < minscale:        # from 50 to 5
        colorjitterScale = random.uniform(-colorRange,colorRange)
        
    if cutout:
        colorjitterScale = -abs(colorjitterScale)
    
    
    # control the texture
    if not control_texture:
        color_mask = np.ones_like(img) * colorjitterScale
        img_jitter = img + color_mask
        img_jitter = img_jitter.clip(0, 255)
        new_img = img * (1-tot_gt_mask) + img_jitter * tot_gt_mask
    else:
        texture_index = random.randint(1,2)
        if texture_index == 1:
            color_mask = np.ones_like(img) * colorjitterScale
            img_jitter = img + color_mask
            img_jitter = img_jitter.clip(0, 255)
            new_img = img * (1-tot_gt_mask) + img_jitter * tot_gt_mask
            
        elif texture_index == 2:
            liver_average = img[np.nonzero(img)].mean()
            img_jitter = randomMap/randomMap.max() * liver_average
            img_jitter = img_jitter.clip(0, 255)
            new_img = img * (1-tot_gt_mask) + img_jitter * tot_gt_mask
    
    combine = img
    combine[tot_gt_mask==1] = 255

    return new_img.astype(np.uint8), tot_gt_mask.astype(np.uint8)
        