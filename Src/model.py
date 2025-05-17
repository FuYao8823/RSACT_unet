from keras.layers import *
from keras.models import Model
def Conv2dT_BN(x, filters, kernel_size, strides=(2, 2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x
def resnet_block(x, n_filter, kernel_size, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, kernel_size, padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, kernel_size, padding="same", strides=1)(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return channel_attention(x)
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])
def Unet_new():
    inpt = Input(shape=(256, 256, 3))

    conv1 = resnet_block(inpt, 8, (3, 3))
    conv1 = resnet_block(conv1, 8, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv2 = resnet_block(pool1, 16, (3, 3))
    conv2 = resnet_block(conv2, 16, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    conv3 = resnet_block(pool2, 32, (3, 3))
    conv3 = resnet_block(conv3, 32, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    conv4 = resnet_block(pool3, 64, (3, 3))
    conv4 = resnet_block(conv4, 64, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)

    conv5 = resnet_block(pool4, 64, (3, 3))
    conv5 = Dropout(0.5)(conv5)
    conv5 = resnet_block(conv5, 64, (3, 3))
    conv5 = Dropout(0.5)(conv5)
    # print(conv5.shape)

    # convt1 = Conv2dT_BN(conv5, 64, (3, 3))
    convt1 = UpSampling2D()(conv5)
    concat1 = concatenate([conv4, convt1], axis=3)
    concat1 = Dropout(0.5)(concat1)
    conv6 = resnet_block(concat1, 64, (3, 3))
    conv6 = resnet_block(conv6, 64, (3, 3))

    # convt2 = Conv2dT_BN(conv6, 32, (3, 3))
    convt2 = UpSampling2D()(conv6)
    concat2 = concatenate([conv3, convt2], axis=3)
    concat2 = Dropout(0.5)(concat2)
    conv7 = resnet_block(concat2, 32, (3, 3))
    conv7 = resnet_block(conv7, 32, (3, 3))

    # convt3 = Conv2dT_BN(conv7, 16, (3, 3))
    convt3 = UpSampling2D()(conv7)
    concat3 = concatenate([conv2, convt3], axis=3)
    concat3 = Dropout(0.5)(concat3)
    conv8 = resnet_block(concat3, 16, (3, 3))
    conv8 = resnet_block(conv8, 16, (3, 3))

    # convt4 = Conv2dT_BN(conv8, 8, (3, 3))
    convt4 = UpSampling2D()(conv8)
    concat4 = concatenate([conv1, convt4], axis=3)
    concat4 = Dropout(0.5)(concat4)
    conv9 = resnet_block(concat4, 8, (3, 3))
    conv9 = resnet_block(conv9, 8, (3, 3))
    conv9 = Dropout(0.5)(conv9)
    outpt = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv9)

    model = Model(inpt, outpt)
    return model
if __name__ == '__main__':

    model = Unet_new()
    model.summary()