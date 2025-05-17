import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image
import cv2
from Src.model import Unet_new
from Src.Unet import Unet
from Src.SegNet import segnet
import os
import dataloader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    model_name = 'Unet_new'  # 'segnet' or 'Unet' or 'Unet_new'

    with open('./test.txt', 'r') as f:
        data_path_val = f.read().splitlines()

    if model_name == 'segnet':
        theat = .6
        model = segnet((256, 256, 3), 1, output_mode='sigmoid')
        model.load_weights('./weights_SegNet2/weights.hdf5')
    elif model_name == 'Unet':
        theat = .8
        model = Unet()
        model.load_weights('./weights_Unet/weights.hdf5')
    elif model_name == 'Unet_new':
        theat = .6
        model = Unet_new()
        model.load_weights('./weights/weights.hdf5')

    dice, precision, recall = [], [], []
    jaccard = []
    for paths in tqdm.tqdm(data_path_val):
        path_img = str("".join(paths).split(' ')[0][2:-1]).replace('data_Aug', 'data_Aug2')
        path_label = str("".join(paths).split(' ')[1][1:-2]).replace('data_Aug', 'data_Aug2')
        if not os.path.exists(path_img):
            continue
        img = np.array(Image.open(path_img))
        label = np.array(Image.open(path_label.replace('.jpg', '.png')))[..., None]
        img = cv2.resize(img, (256, 256))
        label = cv2.resize(label, (256, 256))
        label[label > 0] = 1
        img = dataloader.onezero(img)
        pre = model.predict(img[None], verbose=False)[0, ..., 0]
        precision.append(np.sum((pre > theat) * label) / (np.sum(pre > theat) + 1e-3))
        recall.append(np.sum((pre > theat) * label) / (np.sum(label) + 1e-3))
        dice.append(2 * np.sum((pre > theat) * label) / (np.sum(label) + np.sum(pre > theat) + 1e-3))
        jaccard.append(
            np.sum((pre > theat) * label) / ((np.sum(label) + np.sum(pre > theat) + 1e-3) - np.sum((pre > theat) * label)))
    print(model_name + '\n--------------------------')
    print('Precision is {}'.format(np.mean(precision)))
    print('Recall is {}'.format(np.mean(recall)))
    print('Dice is {}'.format(np.mean(dice)))
    print('Jaccard is {}'.format(np.mean(jaccard)))
#
