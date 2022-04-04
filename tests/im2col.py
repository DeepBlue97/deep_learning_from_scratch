import sys
import os
sys.path.append(os.getcwd())  # 使得可以找到open目录
from open.common.util import im2col


import numpy as np
import cv2

img = cv2.imread(r'D:\workspace\deep_learning\data\g\mmexport1638545352010.jpg')
imgs = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
k = np.array([
    [0.01, 0.1, 0.01],
    [0.01, 0.1, 0.01],
    [0.01, 0.1, 0.01],
])
k = np.stack((k,k,k))
k = np.stack((k,k,k))

col = im2col(input_data=imgs, filter_h=3, filter_w=3, stride=1, pad=0)

print('kernel shape: ', k.shape)  # (3, 3, 3, 3) cv2.resize(img, (10, 14))
# o = conv2d(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), k)
# o = conv2d(np.expand_dims(np.transpose(cv2.resize(img, (108, 144)), (2, 0, 1)), axis=0), k)
# o = conv2d(np.expand_dims(np.transpose(cv2.resize(img, (108*5, 144*5)), (2, 0, 1)), axis=0), k)

# cv2.imwrite('out.jpg', o.squeeze(axis=0).transpose(1,2,0))
print('end')
