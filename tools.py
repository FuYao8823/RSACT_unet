import os

import matplotlib.pyplot as plt
from skimage import io
import SimpleITK
import imgaug.augmenters as iaa  # 导入iaa
import cv2
from PIL import Image
import numpy as np


def Aug():
    seq = iaa.Sequential([
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        # iaa.CropAndPad(px=(-10, 0), percent=None, pad_mode='constant', pad_cval=0, keep_size=True),
        iaa.Affine(scale=(0.92, 1), translate_percent=(0, 0.1), rotate=(-60, 60), cval=0, mode='constant'),
        # iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0),
        # iaa.ElasticTransformation(alpha=(0, 20), sigma=(4.0, 6.0))
    ])
    path_in = './data1'
    path_img = path_in + '/CT'
    path_label = path_in + '/Label'
    path_out = './data_Aug2'
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    if not os.path.exists(path_out + '/CT'):
        os.mkdir(path_out + '/CT')
    if not os.path.exists(path_out + '/Label'):
        os.mkdir(path_out + '/Label')
    number = 0
    for file in os.listdir(path_img):
        img = cv2.imread(path_img + '/' + file)[None]
        img = img.astype(np.float32)

        label = cv2.imread(path_label + '/' + file)
        # label = label.astype(np.int32)
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)[None, ..., None]

        for j in range(10):
            images_aug, segmaps_aug = seq(images=img, segmentation_maps=label)
            # print(segmaps_aug[0].shape)
            # l = cv2.cvtColor(segmaps_aug[0].astype(np.uint8), cv2.COLOR_RGB2GRAY)
            io.imsave(path_out + '/CT/' + str(number) + '.jpg', images_aug[0].astype(np.float32))
            io.imsave(path_out + '/Label/' + str(number) + '.png', (segmaps_aug[0] > 200).astype(np.int8))
            number += 1


def nii2png():
    path_in = '/media/ubuntu/black/zhirui/肺炎分割/tr_im.nii'
    path_seg_in = '/media/ubuntu/black/zhirui/肺炎分割/tr_mask.nii'
    path_out = './data1'
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    imgs = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path_in))
    Segs = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path_seg_in))
    for i in range(len(imgs)):
        if not os.path.exists(path_out + '/CT'):
            os.mkdir(path_out + '/CT')
        if not os.path.exists(path_out + '/Label'):
            os.mkdir(path_out + '/Label')
        io.imsave(path_out + '/CT/' + str(i) + '.jpg', imgs[i])
        Segs[i][Segs[i] == 3] = 0
        io.imsave(path_out + '/Label/' + str(i) + '.jpg', Segs[i] >= 1)


if __name__ == '__main__':
    Aug()

    # img = np.array(Image.open('./data_Aug/Label/6.png'))
    # # print(img.max())
    # img[img > 0] = 1
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img, 'gray')
    # plt.show()
