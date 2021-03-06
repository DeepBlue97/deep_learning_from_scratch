# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0  # 小于0的地方梯度为0，其它地方原样输出
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        # 根据求导公式可得出
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W  # (100, 10) or (4320, 100)
        self.b = b  # (10,)

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分 权重、偏置参数的微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応 张量对应
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x  # (100, 100) or (100, 4320)

        out = np.dot(self.x, self.W) + self.b  # (100, 10) or (100, 100)

        return out

    def backward(self, dout):  # (100, 10)
        dx = np.dot(dout, self.W.T)  # (100, 10).dot((10, 100)) == (100, 100)
        self.dW = np.dot(self.x.T, dout)  # (100, 100).dot((100, 10)) == (100, 10)
        self.db = np.sum(dout, axis=0)  # (10,)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す 恢复输入数据的形状（テンソル対応）张量对应
        return dx  # (100, 10) or (4320, 100)


class Softmax:
    def __init__(self):
        self.y = None
        self.t = None
        self.x = None

    def forward(self, x, t):
        self.t = t
        self.x = x
        self.y = softmax(x)
        return self.y

    def backward(self, dout):
        """
        分两种情况求偏导，可得到梯度为：
        当输入输出标号相同时：si * (1 - si)
        当输入输出标号不同时：-si * sj
        """
        batch_size = self.t.shape[0]
        # jacobian.shape == (batch, 输出length, 输入length)
        jacobian = np.zeros(batch_size, self.y.shape[1], self.x.shape[1])
        # 梯度矩阵赋值
        for bi in range(batch_size):
            for ri in range(jacobian.shape[1]):
                for ci in range(jacobian.shape[2]):
                    if ri == ci:  # 在角线上
                        jacobian[bi][ri][ci] = self.x[bi][ci] * (1 - self.x[bi][ci])
                    else:
                        jacobian[bi][ri][ci] = -self.x[bi][ci] * self.y[bi][ri]
        # # 对角线赋值
        # for bi in jacobian:
        #     for diagi in range(jacobian.shape[1]):
        #         jacobian[bi, diagi, diagi] = self.x[bi, diagi, diagi] * (1 - self.x[bi, diagi, diagi])

        # dx = np.dot(dout, jacobian.T) / batch_size
        dx = np.zeros((batch_size, self.x.shape[1]))
        for bi in range(batch_size):
            dx[bi] = dout.dot(jacobian[bi])

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        以下梯度公式有严格的推理过程，最终结论为si-yi（输入减去标签值）
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx: np.ndarray = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # 使得正确标号对应的输入其偏导为负（使得该处输出值越大损失值越小）
            dx: np.ndarray = dx / batch_size  # 减小偏导的绝对值的简便方法

        return dx  # (100, 10)


class Dropout:
    """
    https://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    https://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose((1, 0)).reshape((FN, C, FH, FW))

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h  # 2
        self.pool_w = pool_w  # 2
        self.stride = stride  # 2
        self.pad = pad  # 0

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)  # (14400, 120)
        col = col.reshape(-1, self.pool_h * self.pool_w)  # (432000, 4)

        arg_max = np.argmax(col, axis=1)  # (432000,) 保存最大值的索引
        out = np.max(col, axis=1)  # (432000,) 保存最大值
        out = out.reshape((N, out_h, out_w, C)).transpose(0, 3, 1, 2)  # (100, 30, 12, 12)

        self.x = x  # (100, 30, 24, 24)
        self.arg_max = arg_max  # ↑

        return out  # (100, 30, 12, 12)

    def backward(self, dout):  # (100, 30, 12, 12)
        dout = dout.transpose(0, 2, 3, 1)  # (100, 12, 12, 30)

        pool_size = self.pool_h * self.pool_w  # 2*2 = 4
        dmax = np.zeros((dout.size, pool_size))  # (432000, 4)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()  # (432000,)
        dmax = dmax.reshape(dout.shape + (pool_size,))  # (100, 12, 12, 30, 4)

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)  # (14400, 120) = 1728000
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)  # (100, 30, 24, 24) = 1728000

        return dx
