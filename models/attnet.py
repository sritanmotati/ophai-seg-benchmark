from __future__ import absolute_import

from utils.keras_unet_collection import *

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import tensorflow as tf
from utils.data_utils import *
from utils.losses import *


def UNET_left(X, channel, kernel_size=3, stack_num=2, activation='ReLU',
              pool=True, batch_norm=False, name='left0'):
    pool_size = 2

    X = encode_layer(X, channel, pool_size, pool, activation=activation,
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv'.format(name))

    return X


def UNET_right(X, X_list, channel, kernel_size=3,
               stack_num=2, activation='ReLU',
               unpool=True, batch_norm=False, concat=True, name='right0'):

    pool_size = 2

    X = decode_layer(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation,
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))
    if concat:
        # <--- *stacked convolutional can be applied here
        X = concatenate([X, ] + X_list, axis=3, name=name + '_concat')

    # Stacked convolutions after concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name=name + '_conv_after_concat')

    return X


def UNET_att_right(X, X_left, channel, att_channel, kernel_size=3, stack_num=2,
                   activation='ReLU', atten_activation='ReLU', attention='add',
                   unpool=True, batch_norm=False, name='right0'):

    pool_size = 2

    X = decode_layer(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    X_left = attention_gate(X=X_left, g=X, channel=att_channel, activation=atten_activation,
                            attention=attention, name='{}_att'.format(name))

    # Tensor concatenation
    H = concatenate([X, X_left], axis=-1, name='{}_concat'.format(name))

    # stacked linear convolutional layers after concatenation
    H = CONV_stack(H, channel, kernel_size, stack_num=stack_num, activation=activation,
                   batch_norm=batch_norm, name='{}_conv_after_concat'.format(name))

    return H


def att_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2,
                     activation='ReLU', atten_activation='ReLU', attention='add', batch_norm=False, pool=True,
                     unpool=True,
                     backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='attunet'):

    activation_func = eval(activation)

    depth_ = len(filter_num)
    X_skip = []

    # no backbone cases
    if backbone is None:
        X = input_tensor
        # downsampling blocks
        X = CONV_stack(X, filter_num[0], stack_num=stack_num_down, activation=activation,
                       batch_norm=batch_norm, name='{}_down0'.format(name))
        X_skip.append(X)

        for i, f in enumerate(filter_num[1:]):
            X = UNET_left(X, f, stack_num=stack_num_down, activation=activation, pool=pool,
                          batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
            X_skip.append(X)

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor, ])
            depth_encode = len(X_skip)

        # for other backbones
        else:
            backbone_ = backbone_zoo(backbone, weights, input_tensor, depth_ - 1, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_skip = backbone_([input_tensor, ])
            depth_encode = len(X_skip) + 1

        # extra conv2d blocks are applied
        # if downsampling levels of a backbone < user-specified downsampling levels
        if depth_encode < depth_:

            # begins at the deepest available tensor
            X = X_skip[-1]

            # extra downsamplings
            for i in range(depth_ - depth_encode):
                i_real = i + depth_encode

                X = UNET_left(X, filter_num[i_real], stack_num=stack_num_down, activation=activation, pool=pool,
                              batch_norm=batch_norm, name='{}_down{}'.format(name, i_real + 1))
                X_skip.append(X)

    # reverse indexing encoded feature maps
    X_skip = X_skip[::-1]
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    depth_decode = len(X_decode)

    # reverse indexing filter numbers
    filter_num_decode = filter_num[:-1][::-1]

    for i in range(depth_decode):
        f = filter_num_decode[i]

        X = UNET_att_right(X, X_decode[i], f, att_channel=f // 2, stack_num=stack_num_up,
                           activation=activation, atten_activation=atten_activation, attention=attention,
                           unpool=unpool, batch_norm=batch_norm, name='{}_up{}'.format(name, i))

    # if tensors for concatenation is not enough
    # then use upsampling without concatenation
    if depth_decode < depth_ - 1:
        for i in range(depth_ - depth_decode - 1):
            i_real = i + depth_decode
            X = UNET_right(X, None, filter_num_decode[i_real], stack_num=stack_num_up, activation=activation,
                           unpool=unpool, batch_norm=batch_norm, concat=False, name='{}_up{}'.format(name, i_real))
    return X


def attnet(input_size, n_labels, filter_num=[64,128,256,512], stack_num_down=2, stack_num_up=2, activation='ReLU',
                atten_activation='ReLU', attention='add', output_activation='Softmax', batch_norm=False, pool=True,
                unpool=True,
                backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='attunet'):

    # one of the ReLU, LeakyReLU, PReLU, ELU
    activation_func = eval(activation)

    if backbone is not None:
        bach_norm_checker(backbone, batch_norm)

    IN = Input(input_size)

    # base
    X = att_unet_2d_base(IN, filter_num, stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                         activation=activation, atten_activation=atten_activation, attention=attention,
                         batch_norm=batch_norm, pool=pool, unpool=unpool,
                         backbone=backbone, weights=weights, freeze_backbone=freeze_backbone,
                         freeze_batch_norm=freeze_backbone, name=name)

    # output layer
    OUT = CONV_output(X, n_labels, kernel_size=1, activation=output_activation, name='{}_output'.format(name))

    # functional API model
    model = Model(inputs=[IN, ], outputs=[OUT, ], name='{}_model'.format(name))

    return model


class AttNet:
    def __init__(self, shape, n_classes):
        self.shape = shape
        self.n_classes = n_classes
        self.model = attnet(self.shape, self.n_classes)

    def summary(self):
        self.model.summary()

    def train(self, train_gen, val_gen, train_steps, val_steps, save_path):
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)]
        self.model.compile(optimizer='adam', losss=dice_coef_multi_loss)
        return self.model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=100, validation_data=val_gen, validation_steps=val_steps, callbacks=callbacks).history

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)

    def get_model(self):
        return self.model