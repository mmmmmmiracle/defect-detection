#%%
from segmentation_models_pytorch import Unet, DeepLabV3, DeepLabV3Plus, PAN, PSPNet, Linknet
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

parser = argparse.ArgumentParser()
parser.add_argument('-backbone', type=str, default='resnet18', help='unet backbone')
args = parser.parse_args()
model = args.backbone
# model = 'mobilenet_v2'
log_dir = f'runs/deeplabv3plus/{model}'
# log_dir = f'runs/deeplabv3/{model}'

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
best_threshold = 0.5
num_workers = 2
batch_size = 4
print('best_threshold', best_threshold)
min_size = 0
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
ckpt_path = os.path.join(log_dir, 'model.pth')
device = torch.device("cuda")
# model = Unet(model, encoder_weights=None, classes=1, activation=None)
model = DeepLabV3Plus(model, encoder_weights=None, classes=1, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

#%%
# start prediction
predictions = []
for i, batch in enumerate(tqdm(testset)):
    fnames, images = batch
    batch_preds = torch.sigmoid(model(images.to(device)))
    batch_preds = batch_preds.detach().cpu().numpy()
    for fname, preds in zip(fnames, batch_preds):
        for cls, pred in enumerate(preds):
            pred, num = post_process(pred, best_threshold, min_size)
            rle = mask2rle(pred)
            # name = fname + f"_{cls+1}"
            name = str(fname.item())
            predictions.append([name, rle])

# save predictions to submission.csv
df = pd.DataFrame(predictions, columns=['filename', 'label'])
df.to_csv(f"{log_dir}/flip_gridshuffle.csv", index=False, header=None)

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
