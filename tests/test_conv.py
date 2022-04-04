import sys
import os
sys.path.append(os.getcwd())  # 使得可以找到open目录
from open.common.util import im2col
from open.common.layers import Convolution


import numpy as np
import cv2

img = cv2.imread(r'D:\workspace\deep_learning\data\g\mmexport1638545352010.jpg')
imgs = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
k = np.array([
    [0.01, 0.1, 0.01],
    [0.01, 0.1, 0.01],
    [0.01, 0.1, 0.01],
])
k = np.stack((k, k, k))
k = np.stack((k*0.8, k*0.8, k))
# k = np.expand_dims(k, axis=0)
print('kernel shape: ', k.shape)  # (3, 3, 3, 3) cv2.resize(img, (10, 14))

conv = Convolution(W=k, b=k.shape[0], stride=1, pad=0)

out = conv.forward(imgs)

cv2.imwrite('out.jpg', out.squeeze(axis=0).transpose(1,2,0))
print('end')
