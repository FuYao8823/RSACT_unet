import datetime
import os
from Src.Unet import Unet
from keras.metrics import Precision, Recall
import dataloader
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)


def dice_loss(smooth=1):
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f)
                                               + K.sum(y_pred_f * y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1. - dice_coef(y_true, y_pred)

    return dice_coef_loss


if __name__ == '__main__':
    if not os.path.exists('./weights_Unet'):
        os.mkdir('./weights_Unet')
    with open('./test.txt', 'r') as f:
        data_path_val = f.read().splitlines()
    with open('./train.txt', 'r') as f:
        data_path_train = f.read().splitlines()
    # data_path_train, data_path_val = dataloader.get_path()

    checkpoint = ModelCheckpoint(monitor='val_accuracy',
                                 filepath="./weights_Unet/weights.hdf5",
                                 verbose=1,
                                 mode='auto',
                                 save_weights_only=True,
                                 save_best_only=True)
    log_dir = "./logs/fit_Unet" + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = Unet()
    model.compile(
        optimizer='adam',
        loss=dice_loss(),
        metrics=['accuracy', Precision(), Recall()]
    )

    model.fit(dataloader.DatasetSequence(data_path_train, 8),
              validation_data=dataloader.DatasetSequence(data_path_val, 8),
              epochs=100,
              callbacks=[checkpoint, tensorboard_callback])
