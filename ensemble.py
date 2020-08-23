#%%
from segmentation_models_pytorch import Unet, DeepLabV3, DeepLabV3Plus, PAN, PSPNet, Linknet
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
warnings.filterwarnings("ignore")
seed = 2020
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm_notebook
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap

#%%
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in
                       (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

#%%
class TestDataset(Dataset):
    '''Dataset for test prediction'''
    def __init__(self, root, df, mean, std):
        self.root = root
        print(self.root)
        # df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.fnames = df['filename'].unique().tolist()
        self.num_samples = len(self.fnames)
        print(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, f'{fname}.png')
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples

#%%
def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((512, 512), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

#%%
sample_submission_path = 'sample_submission.csv'
test_data_folder = "./input/test"

#%%
# initialize test dataloader
best_threshold = 0.469
num_workers = 2
batch_size = 4
print('best_threshold', best_threshold)
min_size = 40
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

#%%
# Initialize mode and load trained weights
# 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4','efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 
# backbones = ['efficientnet-b4', 'efficientnet-b7', 'timm-efficientnet-b8']
backbones = ['efficientnet-b5', 'efficientnet-b6', 'efficientnet-b4', 'efficientnet-b3', 'efficientnet-b2', 'efficientnet-b1', 'efficientnet-b0', 'efficientnet-b7', 'timm-efficientnet-b8']
# backbones = ['efficientnet-b2', 'efficientnet-b1']
models = []
for backbone in backbones:
    log_dir = f'runs/{backbone}'
    ckpt_path = os.path.join(log_dir, 'model.pth')
    device = torch.device("cuda")
    model = Unet(backbone, encoder_weights=None, classes=1, activation=None)
    # model = DeepLabV3Plus(backbone, encoder_weights=None, classes=1, activation=None)
    model.to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    models.append(model)

# backbones2 = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4']
# for backbone in backbones2:
#     log_dir = f'runs/deeplabv3plus/{backbone}'
#     ckpt_path = os.path.join(log_dir, 'model.pth')
#     device = torch.device("cuda")
#     model = DeepLabV3Plus(backbone, encoder_weights=None, classes=1, activation=None)
#     # model = DeepLabV3Plus(backbone, encoder_weights=None, classes=1, activation=None)
#     model.to(device)
#     model.eval()
#     state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
#     model.load_state_dict(state["state_dict"])
#     models.append(model)

class Model:
    def __init__(self, models):
        self.models = models
    
    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x))
        res = torch.stack(res)
        return torch.mean(res, dim=0)
model = Model(models)

#%%
def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        ChannelTranspose()
    ])
    res = A.Compose(res)
    return res

img_folder = 'input/test'

# Different transforms for TTA wrapper
transforms = [
    [],
    [A.HorizontalFlip(p=1)],
    # [A.VerticalFlip(p=1)], 
    # [A.Transpose(p=1)],
    # [A.Resize(768, 768)],
    # [A.Resize(1024, 1024)],
]

transforms = [create_transforms(t) for t in transforms]
datasets = [TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t) for t in transforms]
loaders = [DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False) for d in datasets]

#%%
# thresholds = [0.4,]
# min_area = [40,]
predictions = []
# Iterate over all TTA loaders
total = len(datasets[0])//batch_size
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    # print(len(loaders_batch))
    # break
    preds = []
    image_file = []
    for i, batch in enumerate(loaders_batch):
        # print(type(datasets[i]))
        features = batch['features'].cuda()
        p = torch.sigmoid(model(features))
        # inverse operations for TTA
        if i < 4:
            p = datasets[i].inverse(p)
        else:
            p = p.detach().cpu().numpy()
            # print(p.shape)
            p = np.concatenate([cv2.resize(np.transpose(v, (1, 2, 0)), (512, 512)).reshape(1, 1, 512, 512) for v in p], axis=0)
            # print(p.shape)
            p = torch.from_numpy(p).float().cuda()
        preds.append(p)
        image_file = batch['image_file']
    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()
    
    # Batch post processing
    for p, file in zip(preds, image_file):
        name = os.path.basename(file).split('.')[0]
        # print(name)
        # Image postprocessing
        for i in range(1):
            p_channel = p[i]
            # imageid_classid = file+'_'+str(i+1)
            pred, num = post_process(p_channel, best_threshold, min_size)
            rle = mask2rle(pred)
            predictions.append([name, rle])
    # break
#%%
# start prediction
# predictions = []
# for i, batch in enumerate(tqdm(testset)):
#     fnames, images = batch
#     batch_preds = np.zeros([batch_size, 1, 512, 512])
#     for model in models:
#         batch_preds += torch.sigmoid(model(images.to(device))).detach().cpu().numpy() / len(backbones)
#     # batch_preds = batch_preds.detach().cpu().numpy()
#     for fname, preds in zip(fnames, batch_preds):
#         for cls, pred in enumerate(preds):
#             pred, num = post_process(pred, best_threshold, min_size)
#             rle = mask2rle(pred)
#             # name = fname + f"_{cls+1}"
#             name = str(fname.item())
#             predictions.append([name, rle])

#%%
# save predictions to submission.csv
log_dir = 'ensemble'
df = pd.DataFrame(predictions, columns=['filename', 'label'])
df.filename = df.filename.astype(int)
df = df.sort_values(by='filename')
df.to_csv(f"{log_dir}/flip_gridshuffle.csv", index=False, header=None)
df.head()
#%%
if not os.path.exists(os.path.join(log_dir, 'pics')):
    os.makedirs(os.path.join(log_dir, 'pics'))

def vis(idx, df):
    fname = df.loc[idx].filename
    rle = df.loc[idx].label
    img = cv2.imread(f'input/test/{fname}.png')
    mask = rle2mask(rle, (512, 512))
    plt.figure(figsize=(9, 8))
    plt.subplot(1, 2, 1)
    plt.title(f'img: test set {fname}.png')
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('mask')
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(f'{log_dir}/pics/{fname}.png')
    # plt.show()
    plt.close()

# %%
for i in range(len(df)):
    # if i > 8:
    #     break
    vis(i, df)

# %%
