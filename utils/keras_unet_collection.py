from __future__ import absolute_import
from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
from tensorflow import math
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import warnings
import numpy as np
import random
import pandas as pd
from skimage.color import rgb2gray
from skimage import io
from skimage import img_as_float
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_fill_holes
from tensorflow import expand_dims
from tensorflow.compat.v1 import image
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, UpSampling2D, Conv2DTranspose, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, multiply, add
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.nn import depth_to_space
from tensorflow.image import extract_patches
from tensorflow.keras.layers import Conv2D, Layer, Dense, Embedding, Dropout, Conv2D, LayerNormalization
from tensorflow.keras.activations import softmax
import numpy as np
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from PIL import Image
from tqdm import tqdm
import csv
import math
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D, AveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Concatenate, Dense, Multiply, Lambda, Add
from tensorflow.keras.layers import Activation, Reshape, Dropout, SpatialDropout2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,Concatenate,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.optimizers import Adam

layer_cadidates = {
    'VGG16': ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
    'VGG19': ('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB1': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB2': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB3': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB4': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB5': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB6': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'),
    'EfficientNetB7': (
    'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation',
    'top_activation'), }


def bach_norm_checker(backbone_name, batch_norm):
    '''batch norm checker'''
    if 'VGG' in backbone_name:
        batch_norm_backbone = False
    else:
        batch_norm_backbone = True

    if batch_norm_backbone != batch_norm:
        if batch_norm_backbone:
            param_mismatch = "\n\nBackbone {} uses batch norm, but other layers received batch_norm={}".format(
                backbone_name, batch_norm)
        else:
            param_mismatch = "\n\nBackbone {} does not use batch norm, but other layers received batch_norm={}".format(
                backbone_name, batch_norm)

        warnings.warn(param_mismatch);


def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm):
    '''
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`

    Input
    ----------
        backbone_name: the bakcbone model name. Expected as one of the `tensorflow.keras.applications` class.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0,7]

        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet),
                 or the path to the weights file to be loaded.
        input_tensor: the input tensor
        depth: number of encoded feature maps.
               If four dwonsampling levels are needed, then depth=4.

        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.

    Output
    ----------
        model: a keras backbone model.

    '''

    cadidate = layer_cadidates[backbone_name]

    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max
    # ----- #

    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None, )

    X_skip = []

    for i in range(depth):
        X_skip.append(backbone_.get_layer(cadidate[i]).output)

    model = Model(inputs=[input_tensor, ], outputs=X_skip, name='{}_backbone'.format(backbone_name))

    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)

    return model


def gelu_(X):
    
    return 0.5*X*(1.0 + math.tanh(0.7978845608028654*(X + 0.044715*math.pow(X, 3))))

def snake_(X, beta):

    return X + (1/beta)*math.square(math.sin(beta*X))


class GELU(Layer):
    '''
    Gaussian Error Linear Unit (GELU), an alternative of ReLU
    
    Y = GELU()(X)
    
    ----------
    Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
    
    Usage: use it as a tf.keras.Layer
    
    
    '''
    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def get_config(self):
        config = {'trainable': self.trainable}
        base_config = super(GELU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape

    
class Snake(Layer):
    '''
    Snake activation function $X + (1/b)*sin^2(b*X)$. Proposed to learn periodic targets.
    
    Y = Snake(beta=0.5, trainable=False)(X)
    
    ----------
    Ziyin, L., Hartwig, T. and Ueda, M., 2020. Neural networks fail to learn periodic functions 
    and how to fix it. arXiv preprint arXiv:2006.08195.
    
    '''
    def __init__(self, beta=0.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)

        super(Snake, self).build(input_shape)

    def call(self, inputs, mask=None):
        return snake_(inputs, self.beta_factor)

    def get_config(self):
        config = {'beta': self.get_weights()[0] if self.trainable else self.beta, 'trainable': self.trainable}
        base_config = super(Snake, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    

def decode_layer(X, channel, pool_size, unpool, kernel_size=3, 
                 activation='ReLU', batch_norm=False, name='decode'):
    '''
    An overall decode layer, based on either upsampling or trans conv.
    
    decode_layer(X, channel, pool_size, unpool, kernel_size=3,
                 activation='ReLU', batch_norm=False, name='decode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the decoding factor.
        channel: (for trans conv only) number of convolution filters.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.           
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    * The defaut: `kernel_size=3`, is suitable for `pool_size=2`.
    
    '''
    # parsers
    if unpool is False:
        # trans conv configurations
        bias_flag = not batch_norm
    
    elif unpool == 'nearest':
        # upsample2d configurations
        unpool = True
        interp = 'nearest'
    
    elif (unpool is True) or (unpool == 'bilinear'):
        # upsample2d configurations
        unpool = True
        interp = 'bilinear'
    
    else:
        raise ValueError('Invalid unpool keyword')
        
    if unpool:
        X = UpSampling2D(size=(pool_size, pool_size), interpolation=interp, name='{}_unpool'.format(name))(X)
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
            
        X = Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                            padding='same', name='{}_trans_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
        
    return X

def encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False, name='encode'):
    '''
    An overall encode layer, based on one of the:
    (1) max-pooling, (2) average-pooling, (3) strided conv2d.
    
    encode_layer(X, channel, pool_size, pool, kernel_size='auto', 
                 activation='ReLU', batch_norm=False, name='encode')
    
    Input
    ----------
        X: input tensor.
        pool_size: the encoding factor.
        channel: (for strided conv only) number of convolution filters.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        kernel_size: size of convolution kernels. 
                     If kernel_size='auto', then it equals to the `pool_size`.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    # parsers
    if (pool in [False, True, 'max', 'ave']) is not True:
        raise ValueError('Invalid pool keyword')
        
    # maxpooling2d as default
    if pool is True:
        pool = 'max'
        
    elif pool is False:
        # stride conv configurations
        bias_flag = not batch_norm
    
    if pool == 'max':
        X = MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_maxpool'.format(name))(X)
        
    elif pool == 'ave':
        X = AveragePooling2D(pool_size=(pool_size, pool_size), name='{}_avepool'.format(name))(X)
        
    else:
        if kernel_size == 'auto':
            kernel_size = pool_size
        
        # linear convolution with strides
        X = Conv2D(channel, kernel_size, strides=(pool_size, pool_size), 
                   padding='valid', use_bias=bias_flag, name='{}_stride_conv'.format(name))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_bn'.format(name))(X)
            
        # activation
        if activation is not None:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X

def attention_gate(X, g, channel,  
                   activation='ReLU', 
                   attention='add', name='att'):
    '''
    Self-attention gate modified from Oktay et al. 2018.
    
    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')
    
    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X_att: output tensor.
    
    '''
    activation_func = eval(activation)
    attention_func = eval(attention)
    
    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)
    
    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)
    
    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))
    
    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)
    
    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #
    
    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)
    
    # multiplicative attention masking
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))
    
    return X_att

def CONV_stack(X, channel, kernel_size=3, stack_num=2, 
               dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack'):
    '''
    Stacked convolutional layers:
    (Convolutional layer --> batch normalization --> Activation)*stack_num
    
    CONV_stack(X, channel, kernel_size=3, stack_num=2, dilation_rate=1, activation='ReLU', 
               batch_norm=False, name='conv_stack')
    
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked Conv2D-BN-Activation layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
        
    '''
    
    bias_flag = not batch_norm
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        activation_func = eval(activation)
        
        # linear convolution
        X = Conv2D(channel, kernel_size, padding='same', use_bias=bias_flag, 
                   dilation_rate=dilation_rate, name='{}_{}'.format(name, i))(X)
        
        # batch normalization
        if batch_norm:
            X = BatchNormalization(axis=3, name='{}_{}_bn'.format(name, i))(X)
        
        # activation
        activation_func = eval(activation)
        X = activation_func(name='{}_{}_activation'.format(name, i))(X)
        
    return X

def Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv'):
    '''
    Stacked convolutional layers with residual path.
     
    Res_CONV_stack(X, X_skip, channel, res_num, activation='ReLU', batch_norm=False, name='res_conv')
     
    Input
    ----------
        X: input tensor.
        X_skip: the tensor that does go into the residual path 
                can be a copy of X (e.g., the identity block of ResNet).
        channel: number of convolution filters.
        res_num: number of convolutional layers within the residual path.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''  
    X = CONV_stack(X, channel, kernel_size=3, stack_num=res_num, dilation_rate=1, 
                   activation=activation, batch_norm=batch_norm, name=name)

    X = add([X_skip, X], name='{}_add'.format(name))
    
    activation_func = eval(activation)
    X = activation_func(name='{}_add_activation'.format(name))(X)
    
    return X

def Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1, activation='ReLU', batch_norm=False, name='sep_conv'):
    '''
    Depthwise separable convolution with (optional) dilated convolution kernel and batch normalization.
    
    Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, dilation_rate=1, activation='ReLU', batch_norm=False, name='sep_conv')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked depthwise-pointwise layers.
        dilation_rate: optional dilated convolution kernel.
        activation: one of the `tensorflow.keras.layers` interface, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''
    
    activation_func = eval(activation)
    bias_flag = not batch_norm
    
    for i in range(stack_num):
        X = DepthwiseConv2D(kernel_size, dilation_rate=dilation_rate, padding='same', 
                            use_bias=bias_flag, name='{}_{}_depthwise'.format(name, i))(X)
        
        if batch_norm:
            X = BatchNormalization(name='{}_{}_depthwise_BN'.format(name, i))(X)

        X = activation_func(name='{}_{}_depthwise_activation'.format(name, i))(X)

        X = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_{}_pointwise'.format(name, i))(X)
        
        if batch_norm:
            X = BatchNormalization(name='{}_{}_pointwise_BN'.format(name, i))(X)

        X = activation_func(name='{}_{}_pointwise_activation'.format(name, i))(X)
    
    return X

def ASPP_conv(X, channel, activation='ReLU', batch_norm=True, name='aspp'):
    '''
    Atrous Spatial Pyramid Pooling (ASPP).
    
    ASPP_conv(X, channel, activation='ReLU', batch_norm=True, name='aspp')
    
    ----------
    Wang, Y., Liang, B., Ding, M. and Li, J., 2019. Dense semantic labeling 
    with atrous spatial pyramid pooling and decoder for high-resolution remote 
    sensing imagery. Remote Sensing, 11(1), p.20.
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        activation: one of the `tensorflow.keras.layers` interface, e.g., ReLU.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    * dilation rates are fixed to `[6, 9, 12]`.
    '''
    
    activation_func = eval(activation)
    bias_flag = not batch_norm

    shape_before = X.get_shape().as_list()
    b4 = GlobalAveragePooling2D(name='{}_avepool_b4'.format(name))(X)
    
    b4 = expand_dims(expand_dims(b4, 1), 1, name='{}_expdim_b4'.format(name))
    
    b4 = Conv2D(channel, 1, padding='same', use_bias=bias_flag, name='{}_conv_b4'.format(name))(b4)
    
    if batch_norm:
        b4 = BatchNormalization(name='{}_conv_b4_BN'.format(name))(b4)
        
    b4 = activation_func(name='{}_conv_b4_activation'.format(name))(b4)
    
    # <----- tensorflow v1 resize.
    b4 = Lambda(lambda X: image.resize(X, shape_before[1:3], method='bilinear', align_corners=True), 
                name='{}_resize_b4'.format(name))(b4)
    
    b0 = Conv2D(channel, (1, 1), padding='same', use_bias=bias_flag, name='{}_conv_b0'.format(name))(X)

    if batch_norm:
        b0 = BatchNormalization(name='{}_conv_b0_BN'.format(name))(b0)
        
    b0 = activation_func(name='{}_conv_b0_activation'.format(name))(b0)
    
    # dilation rates are fixed to `[6, 9, 12]`.
    b_r6 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=6, batch_norm=True, name='{}_sepconv_r6'.format(name))
    b_r9 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=9, batch_norm=True, name='{}_sepconv_r9'.format(name))
    b_r12 = Sep_CONV_stack(X, channel, kernel_size=3, stack_num=1, activation='ReLU', 
                        dilation_rate=12, batch_norm=True, name='{}_sepconv_r12'.format(name))
    
    return concatenate([b4, b0, b_r6, b_r9, b_r12])

def CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output'):
    '''
    Convolutional layer with output activation.
    
    CONV_output(X, n_labels, kernel_size=1, activation='Softmax', name='conv_output')
    
    Input
    ----------
        X: input tensor.
        n_labels: number of classification label(s).
        kernel_size: size of 2-d convolution kernels. Default is 1-by-1.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interface or 'Sigmoid'.
                    Default option is 'Softmax'.
                    if None is received, then linear activation is applied.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''
    
    X = Conv2D(n_labels, kernel_size, padding='same', use_bias=True, name=name)(X)
    
    if activation:
        
        if activation == 'Sigmoid':
            X = Activation('sigmoid', name='{}_activation'.format(name))(X)
            
        else:
            activation_func = eval(activation)
            X = activation_func(name='{}_activation'.format(name))(X)
            
    return X



def _crps_tf(y_true, y_pred, factor=0.05):
    
    '''
    core of (pseudo) CRPS loss.
    
    y_true: two-dimensional arrays
    y_pred: two-dimensional arrays
    factor: importance of std term
    '''
    
    # mean absolute error
    mae = K.mean(tf.abs(y_pred - y_true))
    
    dist = tf.math.reduce_std(y_pred)
    
    return mae - factor*dist

def crps2d_tf(y_true, y_pred, factor=0.05):
    
    '''
    (Experimental)
    An approximated continuous ranked probability score (CRPS) loss function:
    
        CRPS = mean_abs_err - factor * std
        
    * Note that the "real CRPS" = mean_abs_err - mean_pairwise_abs_diff
    
     Replacing mean pairwise absolute difference by standard deviation offers
     a complexity reduction from O(N^2) to O(N*logN) 
    
    ** factor > 0.1 may yield negative loss values.
    
    Compatible with high-level Keras training methods
    
    Input
    ----------
        y_true: training target with shape=(batch_num, x, y, 1)
        y_pred: a forward pass with shape=(batch_num, x, y, 1)
        factor: relative importance of standard deviation term.
        
    '''
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    batch_num = y_pred.shape.as_list()[0]
    
    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_tf(y_true[i, ...], y_pred[i, ...], factor=factor)
        
    return crps_out/batch_num


def _crps_np(y_true, y_pred, factor=0.05):
    
    '''
    Numpy version of _crps_tf.
    '''
    
    # mean absolute error
    mae = np.nanmean(np.abs(y_pred - y_true))
    dist = np.nanstd(y_pred)
    
    return mae - factor*dist

def crps2d_np(y_true, y_pred, factor=0.05):
    
    '''
    (Experimental)
    Nunpy version of `crps2d_tf`.
    
    Documentation refers to `crps2d_tf`.
    '''
    
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    
    batch_num = len(y_pred)
    
    crps_out = 0
    for i in range(batch_num):
        crps_out += _crps_np(y_true[i, ...], y_pred[i, ...], factor=factor)
        
    return crps_out/batch_num

# ========================= #
# Dice loss and variants

def dice_coef(y_true, y_pred, const=K.epsilon()):
    '''
    Sørensen–Dice coefficient for 2-d samples.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    
    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)
    
    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const)/(2.0 * true_pos + false_pos + false_neg)
    
    return coef_val

def dice(y_true, y_pred, const=K.epsilon()):
    '''
    Sørensen–Dice Loss.
    
    dice(y_true, y_pred, const=K.epsilon())
    
    Input
    ----------
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    loss_val = 1 - dice_coef(y_true, y_pred, const=const)
    
    return loss_val

# ========================= #
# Tversky loss and variants

def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Weighted Sørensen–Dice coefficient.
    
    Input
    ----------
        y_true, y_pred: predicted outputs and targets.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    
    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    
    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)
    
    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + const)
    
    return coef_val

def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    '''
    Tversky Loss.
    
    tversky(y_true, y_pred, alpha=0.5, const=K.epsilon())
    
    ----------
    Hashemi, S.R., Salehi, S.S.M., Erdogmus, D., Prabhu, S.P., Warfield, S.K. and Gholipour, A., 2018. 
    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks. 
    arXiv preprint arXiv:1803.11078.
    
    Input
    ----------
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases.
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)
    
    return loss_val

def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=K.epsilon()):
    
    '''
    Focal Tversky Loss (FTL)
    
    focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    
    ----------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved 
    attention u-net for lesion segmentation. In 2019 IEEE 16th International Symposium on Biomedical Imaging 
    (ISBI 2019) (pp. 683-687). IEEE.
    
    ----------
    Input
        alpha: tunable parameter within [0, 1]. Alpha handles imbalance classification cases 
        gamma: tunable parameter within [1, 3].
        const: a constant that smooths the loss gradient and reduces numerical instabilities.
        
    '''
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    # (Tversky loss)**(1/gamma) 
    loss_val = tf.math.pow((1-tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)
    
    return loss_val

# ========================= #
# MS-SSIM

def ms_ssim(y_true, y_pred, **kwargs):
    """
    Multiscale structural similarity (MS-SSIM) loss.
    
    ms_ssim(y_true, y_pred, **tf_ssim_kw)
    
    ----------
    Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. Multiscale structural similarity for image quality assessment. 
    In The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). Ieee.
    
    ----------
    Input
        kwargs: keywords of `tf.image.ssim_multiscale`
                https://www.tensorflow.org/api_docs/python/tf/image/ssim_multiscale
                
        *Issues of `tf.image.ssim_multiscale`refers to:
                https://stackoverflow.com/questions/57127626/error-in-calculation-of-inbuilt-ms-ssim-function-in-tensorflow
    
    """
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, **kwargs)
        
    return 1 - tf_ms_ssim

# ======================== #

def iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32):
    
    """
    Inersection over Union (IoU) and generalized IoU coefficients for bounding boxes.
    
    iou_box_coef(y_true, y_pred, mode='giou', dtype=tf.float32)
    
    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. 
    Generalized intersection over union: A metric and a loss for bounding box regression. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).
    
    ----------
    Input
        y_true: the target bounding box. 
        y_pred: the predicted bounding box.
        
        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.
        
        dtype: the data type of input tensors.
               Default is tf.float32.

    """
    
    zero = tf.convert_to_tensor(0.0, dtype)
    
    # subtrack bounding box coords
    ymin_true, xmin_true, ymax_true, xmax_true = tf.unstack(y_true, 4, axis=-1)
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = tf.unstack(y_pred, 4, axis=-1)
    
    # true area
    w_true = tf.maximum(zero, xmax_true - xmin_true)
    h_true = tf.maximum(zero, ymax_true - ymin_true)
    area_true = w_true * h_true
    
    # pred area
    w_pred = tf.maximum(zero, xmax_pred - xmin_pred)
    h_pred = tf.maximum(zero, ymax_pred - ymin_pred)
    area_pred = w_pred * h_pred
    
    # intersections
    intersect_ymin = tf.maximum(ymin_true, ymin_pred)
    intersect_xmin = tf.maximum(xmin_true, xmin_pred)
    intersect_ymax = tf.minimum(ymax_true, ymax_pred)
    intersect_xmax = tf.minimum(xmax_true, xmax_pred)
    
    w_intersect = tf.maximum(zero, intersect_xmax - intersect_xmin)
    h_intersect = tf.maximum(zero, intersect_ymax - intersect_ymin)
    area_intersect = w_intersect * h_intersect
    
    # IoU
    area_union = area_true + area_pred - area_intersect
    iou = tf.math.divide_no_nan(area_intersect, area_union)
    
    if mode == "iou":
        
        return iou
    
    else:
        
        # encolsed coords
        enclose_ymin = tf.minimum(ymin_true, ymin_pred)
        enclose_xmin = tf.minimum(xmin_true, xmin_pred)
        enclose_ymax = tf.maximum(ymax_true, ymax_pred)
        enclose_xmax = tf.maximum(xmax_true, xmax_pred)
        
        # enclosed area
        w_enclose = tf.maximum(zero, enclose_xmax - enclose_xmin)
        h_enclose = tf.maximum(zero, enclose_ymax - enclose_ymin)
        area_enclose = w_enclose * h_enclose
        
        # generalized IoU
        giou = iou - tf.math.divide_no_nan((area_enclose - area_union), area_enclose)

        return giou

def iou_box(y_true, y_pred, mode='giou', dtype=tf.float32):
    """
    Inersection over Union (IoU) and generalized IoU losses for bounding boxes. 
    
    iou_box(y_true, y_pred, mode='giou', dtype=tf.float32)
    
    ----------
    Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. 
    Generalized intersection over union: A metric and a loss for bounding box regression. 
    In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 658-666).
    
    ----------
    Input
        y_true: the target bounding box. 
        y_pred: the predicted bounding box.
        
        Elements of a bounding box should be organized as: [y_min, x_min, y_max, x_max].

        mode: 'iou' for IoU coeff (i.e., Jaccard index);
              'giou' for generalized IoU coeff.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    
    y_true = tf.cast(y_true, dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    return 1 - iou_box_coef(y_true, y_pred, mode=mode, dtype=dtype)


def iou_seg(y_true, y_pred, dtype=tf.float32):
    """
    Inersection over Union (IoU) loss for segmentation maps. 
    
    iou_seg(y_true, y_pred, dtype=tf.float32)
    
    ----------
    Rahman, M.A. and Wang, Y., 2016, December. Optimizing intersection-over-union in deep neural networks for 
    image segmentation. In International symposium on visual computing (pp. 234-244). Springer, Cham.
    
    ----------
    Input
        y_true: segmentation targets, c.f. `keras.losses.categorical_crossentropy`
        y_pred: segmentation predictions.
        
        dtype: the data type of input tensors.
               Default is tf.float32.
        
    """

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.cast(y_pred, dtype)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    area_intersect = tf.reduce_sum(tf.multiply(y_true_pos, y_pred_pos))
    
    area_true = tf.reduce_sum(y_true_pos)
    area_pred = tf.reduce_sum(y_pred_pos)
    area_union = area_true + area_pred - area_intersect
    
    return 1-tf.math.divide_no_nan(area_intersect, area_union)

# ========================= #
# Semi-hard triplet

def triplet_1d(y_true, y_pred, N, margin=5.0):
    
    '''
    (Experimental)
    Semi-hard triplet loss with one-dimensional vectors of anchor, positive, and negative.
    
    triplet_1d(y_true, y_pred, N, margin=5.0)
    
    Input
    ----------
        y_true: a dummy input, not used within this function. Appeared as a requirment of tf.keras.loss function format.
        y_pred: a single pass of triplet training, with `shape=(batch_num, 3*embeded_vector_size)`.
                i.e., `y_pred` is the ordered and concatenated anchor, positive, and negative embeddings.
        N: Size (dimensions) of embedded vectors
        margin: a positive number that prevents negative loss.
        
    '''
    
    # anchor sample pair separations.
    Embd_anchor = y_pred[:, 0:N]
    Embd_pos = y_pred[:, N:2*N]
    Embd_neg = y_pred[:, 2*N:]
    
    # squared distance measures
    d_pos = tf.reduce_sum(tf.square(Embd_anchor - Embd_pos), 1)
    d_neg = tf.reduce_sum(tf.square(Embd_anchor - Embd_neg), 1)
    loss_val = tf.maximum(0., margin + d_pos - d_neg)
    loss_val = tf.reduce_mean(loss_val)
    
    return loss_val


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    print((2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    # ipdb.set_trace()
    # return 1-dice_coef(y_true, y_pred)
    # return tf.reduce_sum(2*(1-dice_coef(y_true[:,:,175:,:], y_pred[:,:,175:,:])))\
    #  + tf.reduce_sum(1-dice_coef(y_true[:,:,:175,:], y_pred[:,:,:175,:])) 
     #fovea weights twice as the OD
    
    Fovea_loss = 0.7*(1-dice_coef(y_true[:,:,175:,:], y_pred[:,:,175:,:]))
    print(Fovea_loss)
    OD_loss = 0.3*(1-dice_coef(y_true[:,:,:175,:], y_pred[:,:,:175,:]))
    return Fovea_loss

def total_loss(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  mse_loss = mse(y_true, y_pred)
  dice_loss =  dice_coef_loss(y_true, y_pred)
  return 0.7*mse_loss+0.3*dice_loss 


class patch_extract(Layer):
    '''
    Extract patches from the input feature map.
    
    patches = patch_extract(patch_size)(feature_map)
    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        feature_map: a four-dimensional tensor of (num_sample, width, height, channel)
        patch_size: size of split patches (width=height)
        
    Output
    ----------
        patches: a two-dimensional tensor of (num_sample*num_patch, patch_size*patch_size)
                 where `num_patch = (width // patch_size) * (height // patch_size)`
                 
    For further information see: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches
        
    '''
    
    def __init__(self, patch_size, **kwargs):
        super(patch_extract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]
    
    def call(self, images):
        
        batch_size = tf.shape(images)[0]
        
        patches = extract_patches(images=images,
                                  sizes=(1, self.patch_size_x, self.patch_size_y, 1),
                                  strides=(1, self.patch_size_x, self.patch_size_y, 1),
                                  rates=(1, 1, 1, 1), padding='VALID',)
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)
        
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num*patch_num, patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)
        
        return patches
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size,})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class patch_embedding(Layer):
    '''
    Embed patches to tokens.
    
    patches_embed = patch_embedding(num_patch, embed_dim)(pathes)
    
    ----------
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, 
    T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. 
    An image is worth 16x16 words: Transformers for image recognition at scale. 
    arXiv preprint arXiv:2010.11929.
    
    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 
        
    Output
    ----------
        embed: Embedded patches.
    
    For further information see: https://keras.io/api/layers/core_layers/embedding/
    
    '''
    
    def __init__(self, num_patch, embed_dim, **kwargs):
        
        super(patch_embedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed

class patch_merging(tf.keras.layers.Layer):
    '''
    Downsample embedded patches; it halfs the number of patches
    and double the embedded dimensions (c.f. pooling layers).
    
    Input
    ----------
        num_patch: number of patches to be embedded.
        embed_dim: number of embedded dimensions. 
        
    Output
    ----------
        x: downsampled patches.
    
    '''
    def __init__(self, num_patch, embed_dim, name='', **kwargs):
        super(patch_merging, self).__init__(**kwargs)
        
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        
        # A linear transform that doubles the channels 
        self.linear_trans = Dense(2*embed_dim, use_bias=False, name='{}_linear_trans'.format(name))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'name':self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, x):
        
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'
        assert (H % 2 == 0 and W % 2 == 0), '{}-by-{} patches received, they are not even.'.format(H, W)
        
        # Convert the patch sequence to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))
        
        # Downsample
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        
        # Convert to the patch squence
        x = tf.reshape(x, shape=(-1, (H//2)*(W//2), 4*C))
       
        # Linear transform
        x = self.linear_trans(x)

        return x

class patch_expanding(tf.keras.layers.Layer):
    '''
    Upsample embedded patches with a given rate (e.g., x2, x4, x8, ...) 
    the number of patches is increased, and the embedded dimensions are reduced.
    
    Input
    ----------
        num_patch: number of patches.
        embed_dim: number of embedded dimensions.
        upsample_rate: the factor of patches expanding, 
                       e.g., upsample_rate=2 doubles input patches and halfs embedded dimensions.
        return_vector: the indicator of returning a sequence of tokens (return_vector=True)  
                       or two-dimentional, spatially aligned tokens (return_vector=False)
                       
    For further information see: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
    '''

    def __init__(self, num_patch, embed_dim, upsample_rate, return_vector=True, name='patch_expand', **kwargs):
        super(patch_expanding, self).__init__(**kwargs)
        
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.upsample_rate = upsample_rate
        self.return_vector = return_vector
        
        # Linear transformations that doubles the channels 
        self.linear_trans1 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        # 
        self.linear_trans2 = Conv2D(upsample_rate*embed_dim, kernel_size=1, use_bias=False, name='{}_linear_trans1'.format(name))
        self.prefix = name
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
            'upsample_rate': self.upsample_rate,
            'return_vector': self.return_vector,
            'name':self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, x):
        
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        
        assert (L == H * W), 'input feature has wrong size'

        x = tf.reshape(x, (-1, H, W, C))
        
        x = self.linear_trans1(x)
        
        # rearange depth to number of patches
        x = depth_to_space(x, self.upsample_rate, data_format='NHWC', name='{}_d_to_space'.format(self.prefix))
        
        if self.return_vector:
            # Convert aligned patches to a patch sequence
            x = tf.reshape(x, (-1, L*self.upsample_rate*self.upsample_rate, C//2))

        return x
    
# ========== Swin-Transformer only ========== #
# TODO: documentation.

def window_partition(x, window_size):
    
    # Get the static shape of the input tensor
    # (Sample, Height, Width, Channel)
    _, H, W, C = x.get_shape().as_list()
    
    # Subset tensors to patches
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(x, shape=(-1, patch_num_H, window_size, patch_num_W, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    
    # Reshape patches to a patch sequence
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))
    
    return windows

def window_reverse(windows, window_size, H, W, C):
    
    # Reshape a patch sequence to aligned patched 
    patch_num_H = H//window_size
    patch_num_W = W//window_size
    x = tf.reshape(windows, shape=(-1, patch_num_H, patch_num_W, window_size, window_size, C))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    
    # Merge patches to spatial frames
    x = tf.reshape(x, shape=(-1, H, W, C))
    
    return x

def drop_path_(inputs, drop_prob, is_training):
    
    # Bypass in non-training mode
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    input_shape = tf.shape(inputs)
    batch_num = input_shape[0]; rank = len(input_shape)
    
    shape = (batch_num,) + (1,) * (rank - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    path_mask = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * path_mask
    return output

class drop_path(Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(drop_path, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def get_config(self):
        config = super().get_config().copy()
        config.update({'drop_prob': self.drop_prob})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, x, training=None):
        return drop_path_(x, self.drop_prob, training)

class Mlp(tf.keras.layers.Layer):
    def __init__(self, filter_num, drop=0., name='mlp', **kwargs):
        
        super(Mlp, self).__init__(**kwargs)
        
        self.filter_num = filter_num
        self.drop = drop
        
        # MLP layers
        self.fc1 = Dense(filter_num[0], name='{}_mlp_0'.format(name))
        self.fc2 = Dense(filter_num[1], name='{}_mlp_1'.format(name))
        
        # Dropout layer
        self.drop = Dropout(drop)
        
        # GELU activation
        self.activation = tf.keras.activations.gelu
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filter_num': self.filter_num,
            'drop': self.drop,
            'name': self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, x):
        
        # MLP --> GELU --> Drop --> MLP --> Drop
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                 attn_drop=0, proj_drop=0., name='swin_atten', **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        
        self.dim = dim # number of input dimensions
        self.window_size = window_size # size of the attention window
        self.num_heads = num_heads # number of self-attention heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 # query scaling factor
        
        self.prefix = name
        
        # Layers
        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name='{}_attn_qkv'.format(self.prefix))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name='{}_attn_proj'.format(self.prefix))
        self.proj_drop = Dropout(proj_drop)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dim':self.dim, 
            'window_size':self.window_size, 
            'num_heads':self.num_heads, 
            'qkv_bias':self.qkv_bias, 
            'qk_scale':self.qk_scale, 
            'attn_drop':self.attn_drop, 
            'proj_drop':self.proj_drop, 
            'name':self.prefix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def build(self, input_shape):
        
        # zero initialization
        num_window_elements = (2*self.window_size[0] - 1) * (2*self.window_size[1] - 1)
        self.relative_position_bias_table = self.add_weight('{}_attn_pos'.format(self.prefix),
                                                            shape=(num_window_elements, self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)
        
        # Indices of relative positions
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        
        # convert to the tf variable
        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False, name='{}_attn_pos_ind'.format(self.prefix))
        
        self.built = True

    def call(self, x, mask=None):
        
        # Get input tensor static shape
        _, N, C = x.get_shape().as_list()
        head_dim = C//self.num_heads
        
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        
        # Query rescaling
        q = q * self.scale
        
        # multi-headed self-attention
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = (q @ k)
        
        # Shift window
        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(relative_position_bias, shape=(num_window_elements, num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32)
            attn = tf.reshape(attn, shape=(-1, nW, self.num_heads, N, N)) + mask_float
            attn = tf.reshape(attn, shape=(-1, self.num_heads, N, N))
            attn = softmax(attn, axis=-1)
        else:
            attn = softmax(attn, axis=-1)
        
        # Dropout after attention
        attn = self.attn_drop(attn)
        
        # Merge qkv vectors
        x_qkv = (attn @ v)
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, N, C))
        
        # Linear projection
        x_qkv = self.proj(x_qkv)
        
        # Dropout after projection
        x_qkv = self.proj_drop(x_qkv)
        
        return x_qkv

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, 
                 num_mlp=1024, qkv_bias=True, qk_scale=None, mlp_drop=0, attn_drop=0, 
                 proj_drop=0, drop_path_prob=0, name='swin_block', **kwargs):
        
        super(SwinTransformerBlock, self).__init__(**kwargs)
        
        self.dim = dim # number of input dimensions
        self.num_patch = num_patch # number of embedded patches; a tuple of  (heigh, width)
        self.num_heads = num_heads # number of attention heads
        self.window_size = window_size # size of window
        self.shift_size = shift_size # size of window shift
        self.num_mlp = num_mlp # number of MLP nodes
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path_prob = drop_path_prob
        
        self.prefix = name
        
        # Layers
        self.norm1 = LayerNormalization(epsilon=1e-5, name='{}_norm1'.format(self.prefix))
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, name=self.prefix)
        self.drop_path = drop_path(drop_path_prob)
        self.norm2 = LayerNormalization(epsilon=1e-5, name='{}_norm2'.format(self.prefix))
        self.mlp = Mlp([num_mlp, dim], drop=mlp_drop, name=self.prefix)
        
        # Assertions
        assert 0 <= self.shift_size, 'shift_size >= 0 is required'
        assert self.shift_size < self.window_size, 'shift_size < window_size is required'
        
        # <---!!!
        # Handling too-small patch numbers
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dim':self.dim, 
            'num_patch':self.num_patch, 
            'num_heads':self.num_heads, 
            'window_size':self.window_size, 
            'shift_size':self.shift_size, 
            'num_mlp':self.num_mlp,
            'qkv_bias':self.qkv_bias, 
            'qk_scale':self.qk_scale, 
            'mlp_drop':self.mlp_drop, 
            'attn_drop':self.attn_drop, 
            'proj_drop':self.proj_drop, 
            'drop_path_prob':self.drop_path_prob, 
            'name':self.prefix
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.num_patch
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            
            # attention mask
            mask_array = np.zeros((1, H, W, 1))
            
            ## initialization
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)
            
            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False, name='{}_attn_mask'.format(self.prefix))
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.num_patch
        B, L, C = x.get_shape().as_list()
        
        # Checking num_path and tensor sizes
        assert L == H * W, 'Number of patches before and after Swin-MSA are mismatched.'
        
        # Skip connection I (start)
        x_skip = x
        
        # Layer normalization
        x = self.norm1(x)
        
        # Convert to aligned patches
        x = tf.reshape(x, shape=(-1, H, W, C))

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # Window partition 
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size * self.window_size, C))

        # Window-based multi-headed self-attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
            
        # Convert back to the patch sequence
        x = tf.reshape(x, shape=(-1, H*W, C))

        # Drop-path
        ## if drop_path_prob = 0, it will not drop
        x = self.drop_path(x)
        
        # Skip connection I (end)
        x = x_skip +  x
        
        # Skip connection II (start)
        x_skip = x
        
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        
        # Skip connection II (end)
        x = x_skip + x

        return x

class_names = ['AMD', 'NON-AMD']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

IMAGE_SIZE = (512, 512)
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, name=""):
    padding = "SAME" if strides == 1 else "VALID"
    return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "conv")(inputs)

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def create_dir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if os.path.isdir(mypath):
            pass
        else:
            raise


def plot_loss(loss, label, filename, log_dir, acc=None, title='', ylim=None):
    """Plot a loss function and save it in a file."""
    loss = np.array(loss)
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        if acc is None:
            plt.ylim((0, 0.5))
        else:
            plt.ylim((0,1.))

    plt.title(title)
    plt.savefig(os.path.join(log_dir, filename))
    plt.clf()
    plt.close('all')

def resizefile(pathname,dirname,savepathname):
    
    for item in sorted(dirname):
        if item == '.DS_Store':
             continue
        if os.path.isfile(pathname+item):
        
            img = Image.open(pathname+item)
            x,y = img.size
            print(x)
            Ratio.append([x,y])
            f, e = os.path.splitext(item)
            imResize = img.resize((512,512),Image.ANTIALIAS)
            
            imResize.save(savepathname + f + ' resized.jpg', 'JPEG')

def get_dist_maps(coords, shp=(512,512)):
    fx, fy = coords[0]
    

    distance = np.ones(shp)
    distance[fy, fx] = 0
    
    distance = distance_transform_edt(distance)
    distance = distance[:,:,np.newaxis]
    if shp != (512,512):
        distance=resize(1 - distance / np.max(distance), (512,512,3)) ** 7
    else:
        distance = (1 - distance / np.max(distance)) ** 7
    return distance

"""
Iterator to load images from the datasets, and related functions.
"""
def normalize_for_tanh(batch):
    """Make input image values lie between -1 and 1."""
    tanh_batch = batch - np.max(batch)/2.
    tanh_batch /= np.max(batch)/2.
    return tanh_batch

class TwoImageIterator(Iterator):
    """Class to iterate A and B images at the same time, while applying desired
    transformations online."""

    def __init__(self, directory, a_dir_name='A', b_dir_name=None, N=-1,
                 batch_size=32, shuffle=True, seed=None, target_size=(512,512),
                 cspace='rgb', nch_gdt=1,
                 zscore=True, normalize_tanh=False,
                 return_mode='normal', decay=5, dataset='idrid',
                 rotation_range=0., height_shift_range=0., shear_range=0.,
                 width_shift_range=0., zoom_range=0., fill_mode='constant',
                 cval=0., horizontal_flip=False, vertical_flip=False, transform=None):

        """
        Iterate through the image directoriy, apply transformations and return
        distance map calculated on the fly. If b_dir_name is not None, it will
        retrieve the ground truth from the directory.

        Files under the directory A and B will be returned at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator;
        - return_mode: 'normal', 'fnames'. Default: 'normal'
            - 'normal' returns: [batch_a, batch_b]
            - 'fnames' returns: [batch_a, batch_b, files]
        - decay: decay at which to compute de distance map. Default: 5
        - dataset: dataset to load. Can handle Messidor and Idrid. Default: Idrid

        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.a_fnames = sorted(os.listdir(self.a_dir))

        self.b_dir_name = b_dir_name
        if b_dir_name is not None:
            self.b_dir = os.path.join(directory, b_dir_name)
            self.b_fnames = sorted(os.listdir(self.b_dir))

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            self.filenames = self.a_fnames[:N]
        self.N = len(self.a_fnames)

        self.ch_order = 'tf'

        # Preprocess images
        self.cspace = cspace #colorspace

        # Image shape
        self.target_size = target_size
        self.nch_gdt = nch_gdt

        self.nch = len(self.cspace) # for example if grayscale

        #self.select_vessels = select_vessels

        self.img_shape_a = self._get_img_shape(self.target_size, ch=self.nch)
        self.img_shape_b = self._get_img_shape(self.target_size, ch=self.nch_gdt)
       

        if self.ch_order == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2
        else:
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3

        #Normalizations
        self.normalize_tanh = normalize_tanh
        self.zscore = zscore

        # Transformations
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        self.transform = transform

        self.return_mode = return_mode
        self.decay=decay
        self.dataset = dataset

        super(TwoImageIterator, self).__init__(len(self.a_fnames), batch_size,
                                               shuffle, seed)

    def _get_img_shape(self, size, ch=3):

        if self.ch_order == 'tf':
            img_shape = size + (ch,)
        else:
            img_shape = (ch,) + size

        return img_shape

    def _load_img_pair(self, idx):
        """
        Load images and apply pre-processing
        :param idx: index of file to load in the list of names
        :return: aa: image
                 bb: ground truth
        """
        #print(self.a_fnames[276])
        #print(len(self.a_fnames),len(self.b_fnames))
        
        
        aa = cv2.imread(os.path.join(self.a_dir, self.a_fnames[idx]))
        aa = cv2.cvtColor(aa, cv2.COLOR_BGR2RGB)
        bb = cv2.imread(os.path.join(self.b_dir, self.b_fnames[idx]))
        bb = cv2.cvtColor(bb, cv2.COLOR_BGR2RGB)
        #print(self.a_fnames[idx]+" "+self.b_fnames[idx])
        
        if self.nch_gdt == 3:
            # fix for the case when the .png has an alpha channel
            if bb.shape[-1] == 4:
                bb = bb[:,:,:3]
        elif self.nch_gdt == 1:
            # fix for the case when the .png has an alpha channel
            
            if len(bb.shape) == 3:
                bb = rgb2gray(bb)
                bb = bb.reshape((512,512,1))
                
        #if self.select_vessels is True:
            #bb = self.select_vessel_width(bb)
        
        return aa, bb

    def _random_transform(self, a, b, is_batch=False):
        if is_batch is False:
        # a and b are single images, so they don't have image number at index 0
            img_row_index = self.row_index - 1
            img_col_index = self.col_index - 1
            img_channel_index = self.channel_index - 1
        else:
            img_row_index = self.row_index
            img_col_index = self.col_index
            img_channel_index = self.channel_index
        """
        New augumentation implement

        """
        if self.transform is not None:
          a = self.transform(image=a)['image']
          # b = self.transform(image=b)['image'] #no need to transform ground truth img

        return a, b

    def get_dist_maps(self, coords, shp=(512,512)):
        fx, fy = coords[0]
    

        distance = np.ones(shp)
        distance[fy, fx] = 0
    
        distance = distance_transform_edt(distance)
        distance = distance[:,:,np.newaxis]
        if shp != (512,512):
            distance=resize(1 - distance / np.max(distance), (512,512,3)) ** 7
        else:
            distance = (1 - distance / np.max(distance)) ** 7
        return distance

    def next(self):
        """Get the next pair of the sequence."""

        # Lock the iterator when the index is changed.
        with self.lock:
            index_array = next(self.index_generator)
        current_batch_size = len(index_array)
            
        # Initialize the arrays according to the size of the output images
       
        batch_a = np.zeros((current_batch_size,) + self.img_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.img_shape_b[:-1]
                           + (self.nch_gdt,))

        files = []
        ind = []


        # Load images and apply transformations
        for i, j in enumerate(index_array):
            
            if self.a_fnames[j] == ".DS_Store":
                continue
            
            im_id = self.a_fnames[j][:-4]
            a_img, b_img = self._load_img_pair(j)

            #Transform
            a_img, b_img = self._random_transform(a_img, b_img)

            # #NORMALIZE
            if self.zscore is True:
                a_img = (a_img - a_img.mean()) / (a_img.std())        

            batch_a[i] = a_img
            batch_b[i] = b_img

            files.append(self.a_fnames[j])

        # when using tanh activation the inputs must be between [-1 1]
        if self.normalize_tanh is True and self.zscore is False:
            batch_a = normalize_for_tanh(batch_a)
            batch_b = normalize_for_tanh(batch_b)

        if self.return_mode == 'normal':
            return [batch_a, batch_b]

        elif self.return_mode == 'fnames':
            return [batch_a, batch_b, files]

def check_EarlyStop(vloss, tr_loss, patience=5):

    Eopt = np.min(vloss[:-1])
    GL = ((vloss[-1] / Eopt) - 1)
    Pk = (np.sum(tr_loss[-patience:]) / (
            patience * (np.min(tr_loss[-patience:]))))
    PQ = GL / Pk

    # return [PQ, GL, Pk]
    if (GL > 0.):
        if PQ > 0.5:
            return 'early_stop'
        elif Pk < 1.1:
            return 'early_stop'
        else:
            return 'pass'
    else:
        return 'pass'

def dummy_loader(model_path):
    '''
    Load a stored keras model and return its weights.
    
    Input
    ----------
        The file path of the stored keras model.
    
    Output
    ----------
        Weights of the model.
        
    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def image_to_array(filenames, size, channel):
    '''
    Converting RGB images to numpy arrays.
    
    Input
    ----------
        filenames: an iterable of the path of image files
        size: the output size (height == width) of image. 
              Processed through PIL.Image.NEAREST
        channel: number of image channels, e.g. channel=3 for RGB.
        
    Output
    ----------
        An array with shape = (filenum, size, size, channel)
        
    '''
    
    # number of files
    L = len(filenames)
    
    # allocation
    out = np.empty((L, size, size, channel))
    
    # loop over filenames
    if channel == 1:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ..., 0] = np.array(pix)
    else:
        for i, name in enumerate(filenames):
            with Image.open(name) as pixio:
                pix = pixio.resize((size, size), Image.NEAREST)
                out[i, ...] = np.array(pix)[..., :channel]
    return out[:, ::-1, ...]

def shuffle_ind(L):
    '''
    Generating random shuffled indices.
    
    Input
    ----------
        L: an int that defines the largest index
        
    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''
    
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind

def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model
    
    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization    
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model

