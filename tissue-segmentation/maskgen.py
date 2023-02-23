import sys
sys.path.extend(["../.", "."])
from common.wsi_reader import get_reader_impl
import os
import gc
import cv2
import time
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import glob
import scipy.signal
from unet import UNet
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.morphology import remove_small_objects
from run import *
from openslide import OpenSlide

def get_WINDOW_SPLINE_2D(patch_size, effective_window_size, power = 2):
    window_size = effective_window_size
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)

    aug = int(round((patch_size - window_size) / 2.0))
    wind = np.pad(wind, (aug, aug), mode='constant')
    wind = wind[:patch_size]
    wind = np.expand_dims(np.expand_dims(wind, 1), 2)
    wind = wind * wind.transpose(1, 0, 2)
    return wind


slide_path = '/mnt/disks/data-store-2/Coreplus/fdata/T0018.mrxs'
save_folder = '/mnt/disks/data-store-2/Coreplus/fdata/pathprofiler/result'
model = '/mnt/disks/data-store-2/Coreplus/fdata/pathprofiler/model/checkpoint_ts.pth'
mask_magnification = 1.25
gpu_id = '0'
tile_size = 512
batch_size = 4
downsample = 32

unet = UNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = nn.DataParallel(unet).cuda() if torch.cuda.is_available() else nn.DataParallel(unet)
checkpoint = torch.load(model, map_location=device)
net.load_state_dict(checkpoint['state_dict'])

slide = OpenSlide(slide_path)

slide_level_dimensions = (int(np.round(slide.level_dimensions[0][0]/downsample)),
                                  int(np.round(slide.level_dimensions[0][1]/downsample)))
aug = int(round(512 * (1 - 1.0 / 2.0)))
more_borders = ((aug, aug), (aug, aug), (0, 0))
thumbnail = slide.get_thumbnail(slide_level_dimensions)
img = slide.get_thumbnail(slide_level_dimensions)
img = np.asarray(img.convert('RGB'))
padded_img = np.pad(img, pad_width=more_borders, mode='reflect')
test_dataset = SegDataset(padded_img, 512, 2.0)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            shuffle=False)

gc.collect()

# run the model in batches
all_prediction = []
for patches in tqdm(test_loader):
    if torch.cuda.is_available():
        patches = patches.cuda()

    all_prediction += [net(patches).cpu().data.numpy()]

all_prediction = np.concatenate(all_prediction, axis=0)
all_prediction = all_prediction.transpose(0, 2, 3, 1)
patch_size = 512
effective_window_size = 512
power = 2

WINDOW_SPLINE_2D = get_WINDOW_SPLINE_2D(patch_size = tile_size, effective_window_size = tile_size, power=2)
padded_img_size = padded_img.shape
patches = all_prediction
n_dims = patches[0].shape[-1]
img = np.zeros([padded_img_size[0], padded_img_size[1], n_dims], dtype=np.float32)

window_size = 512
step = int(window_size / 2.0)

row_range = range(0, img.shape[0] - 512 + 1, step)
col_range = range(0, img.shape[1] - 512 + 1, step)

for index1, row in enumerate(row_range):
    for index2, col in enumerate(col_range):
        tmp = patches[(index1 * len(col_range)) + index2]
        tmp *= WINDOW_SPLINE_2D

        img[row:row + patch_size, col:col + patch_size, :] = \
            img[row:row + patch_size, col:col + patch_size, :] + tmp

img = img / (2.0 ** 2)
aug = int(round(patch_size * (1 - 1.0 / 2.0)))
ret = img[aug:-aug, aug:-aug, :]
result = np.argmax(ret, axis=2) * 255.0
result = result.astype(np.uint8)
result = np.clip(result, 0, 255).astype(np.uint8)
segmentation = remove_small_objects(result == 255, 50**2)
segmentation = (segmentation*255).astype(np.uint8)
savename = os.path.join(save_folder, 'temp.png')
cv2.imwrite(savename, segmentation)