import os
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import Sequence
import cv2
from PIL import Image
from keras.utils import to_categorical
from utlis import tools
np.random.seed(2023)


def onezero(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img


def read_img(path):
    img = cv2.imread(path)
    # print(path)
    # print(img.shape)
    img, label = tools.img2label(img)
    img = cv2.resize(img, (256, 256))
    label = cv2.resize(label, (256, 256))
    label[label > 0] = 1
    img = onezero(img)
    return img, label[..., None]


def read_batch_data(paths):
    batch_img, batch_label = [], []
    for path in paths:
        path_img = str("".join(path).split(' ')[0][2:-1]).replace('data_Aug', 'data_Aug2')
        path_label = str("".join(path).split(' ')[1][1:-2]).replace('data_Aug', 'data_Aug2')
        img = np.array(Image.open(path_img))
        label = np.array(Image.open(path_label.replace('.jpg', '.png')))[..., None]
        img = cv2.resize(img, (256, 256))
        label = cv2.resize(label, (256, 256))
        label[label > 0] = 1
        img = onezero(img)
        batch_img.append(img)
        batch_label.append(label)
    batch_img = np.array(batch_img)
    batch_label = np.array(batch_label)
    return batch_img, batch_label


def get_class_batch_label(paths):
    out_label = []
    for path in paths:
        label = int(path.split('/')[-2]) - 1
        out_label.append(label)
    out_label = np.array(out_label).reshape((-1, 1))
    out_label = to_categorical(out_label, num_classes=6)
    return out_label


class DatasetSequence(Sequence):
    def __init__(self, set_all, batch_size):
        self.set_all = set_all
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.set_all) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img, batch_seg_label = read_batch_data(self.set_all[idx * self.batch_size: (idx + 1) * self.batch_size])
        # print(batch_img.shape)
        # print(batch_seg_label.shape)
        return batch_img.astype(np.float32), batch_seg_label.astype(np.float32)[..., None]

    def on_epoch_end(self):
        np.random.shuffle(self.set_all)


def get_path():
    path_all = './data_Aug1'
    path_CT = path_all + '/CT'
    path_label = path_all + '/Label'
    data_paths = []
    for file in sorted(os.listdir(path_CT)):
        data_paths.append([path_CT + '/' + file, path_label + '/' + file])
    data_paths = np.array(data_paths)
    arr = np.array_split(data_paths, 10)
    data_path_train, data_path_val = np.concatenate(arr[:-1], axis=0), arr[-1]
    return data_path_train, data_path_val


def get_val_path():
    path_all = './data'
    data_paths = []
    for files in os.listdir(path_all):
        for file in sorted(os.listdir(path_all + '/' + files)):
            if file.split('.')[-1] != 'db':
                data_paths.append(path_all + '/' + files + '/' + file)
    data_paths = np.array(data_paths)
    np.random.shuffle(data_paths)
    data_path_val = np.array_split(data_paths, 10)[-1]
    return data_path_val


if __name__ == '__main__':
    x, y = get_path()

