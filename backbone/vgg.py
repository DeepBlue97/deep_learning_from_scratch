from collections import OrderedDict

import numpy as np

from common.layers import (
    Pooling,
    Convolution,
    Affine,
    Softmax,
)


class VGG11:
    def __init__(self, input_dim=(3, 224, 224), weight_init_std=0.01):
        # weights init
        self.params = {
            'W1': weight_init_std * np.random.randn(64, input_dim[0], 3, 3), 'b1': np.zeros(64),
            'W2': weight_init_std * np.random.randn(128, 64, 3, 3), 'b2': np.zeros(128),
            'W3': weight_init_std * np.random.randn(256, 128, 3, 3), 'b3': np.zeros(256),
            'W4': weight_init_std * np.random.randn(256, 256, 3, 3), 'b4': np.zeros(256),
            'W5': weight_init_std * np.random.randn(512, 256, 3, 3), 'b5': np.zeros(512),
            'W6': weight_init_std * np.random.randn(512, 512, 3, 3), 'b6': np.zeros(512),
            'W7': weight_init_std * np.random.randn(512, 512, 3, 3), 'b7': np.zeros(512),
            'W8': weight_init_std * np.random.randn(512, 512, 3, 3), 'b8': np.zeros(512),
            'W9': weight_init_std * np.random.randn(25088, 4096), 'b9': np.zeros(4096),
            'W10': weight_init_std * np.random.randn(4096, 4096), 'b10': np.zeros(4096),
            'W11': weight_init_std * np.random.randn(4096, 1000), 'b11': np.zeros(1000),
        }

        # layers init
        self.layers = OrderedDict()
        self.layers['conv1'] = Convolution(W=self.params['W1'], b=self.params['b1'], stride=1, pad=1)
        self.layers['maxpooling1'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)  # out 112
        self.layers['conv2'] = Convolution(W=self.params['W2'], b=self.params['b2'], stride=1, pad=1)
        self.layers['maxpooling2'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)  # out 56
        self.layers['conv3'] = Convolution(W=self.params['W3'], b=self.params['b3'], stride=1, pad=1)
        self.layers['conv4'] = Convolution(W=self.params['W4'], b=self.params['b4'], stride=1, pad=1)
        self.layers['maxpooling3'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)  # out 28*28
        self.layers['conv5'] = Convolution(W=self.params['W5'], b=self.params['b5'], stride=1, pad=1)
        self.layers['conv6'] = Convolution(W=self.params['W6'], b=self.params['b6'], stride=1, pad=1)
        self.layers['maxpooling4'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)  # out 14*14
        self.layers['conv7'] = Convolution(W=self.params['W7'], b=self.params['b7'], stride=1, pad=1)
        self.layers['conv8'] = Convolution(W=self.params['W8'], b=self.params['b8'], stride=1, pad=1)
        self.layers['maxpooling5'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)  # out 7*7
        self.layers['FC4096'] = Affine(W=self.params['W9'], b=self.params['b9'])  # out 4096
        self.layers['FC4096'] = Affine(W=self.params['W10'], b=self.params['b10'])  # out 4096
        self.layers['FC1000'] = Affine(W=self.params['W11'], b=self.params['b11'])  # out 1000
        self.layers['softmax'] = Softmax()
