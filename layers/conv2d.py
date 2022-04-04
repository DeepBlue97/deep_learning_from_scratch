import numpy as np

"""
该算法用了较多的循环，但是较为节省资源开销，计算复杂度对于像素个数来说为n
缺点是只用到了单核性能。
"""

def conv2d(IN: np.ndarray, KERNEL: np.ndarray, PADDING: bool | int = 0):
    """
    input:
        IN: np.ndarray(IN_B, IN_C, IN_H, IN_W)
        KERNEL: np.ndarray(KERNEL_B, KERNEL_C=IN_C, KERNEL_H, KERNEL_W)

    output: 
        out: np.ndarray(OUT_B=IN_B, OUT_C=KERNEL_B, OUT_H, OUT_W)

    """
    # 自动设置PADDING量
    if PADDING is True:
        PADDING = int((KERNEL_W - 1) / 2)

    IN_B = IN.shape[0]
    IN_C = IN.shape[1]
    IN_H = IN.shape[2]
    IN_W = IN.shape[3]

    KERNEL_B = KERNEL.shape[0]
    KERNEL_C = KERNEL.shape[1]
    KERNEL_H = KERNEL.shape[2]
    KERNEL_W = KERNEL.shape[3]

    OUT_B = IN_B
    OUT_C = KERNEL_B  # 输出通道数（卷积核批次数）
    OUT_H = IN_H if PADDING is True else IN_H + PADDING*2 - (KERNEL_H - 1)
    OUT_W = IN_W if PADDING is True else IN_W + PADDING*2 - (KERNEL_W - 1)

    assert IN_C == KERNEL_C  # 输入通道数和卷积核通道数相同

    # 定义输出张量
    out = np.zeros([OUT_B, OUT_C, OUT_H, OUT_W])

    # 输入张量PAD之后的张量，真正待卷积的张量
    IN_PADDED_B = IN_B
    IN_PADDED_C = IN_C
    IN_PADDED_H = IN_H + PADDING
    IN_PADDED_W = IN_W + PADDING

    # PADDING后的输入
    IN_PADDED = np.concatenate(
        (
            np.zeros((IN_PADDED_B, IN_PADDED_C, PADDING, IN_PADDED_W)),
            np.concatenate((np.zeros((IN_PADDED_B, IN_PADDED_C, IN_PADDED_H, PADDING)),
                            IN,
                            np.zeros((IN_PADDED_B, IN_PADDED_C, IN_PADDED_H, PADDING))), axis=-1),  # 列维度padding
            np.zeros((IN_PADDED_B, IN_PADDED_C, PADDING, IN_PADDED_W))
        ), axis=-2  # 行维度padding
    )
    # 检查根据IN张量经PADDING后的IN_PADDED张量形状是否正确
    assert IN_PADDED.shape == (
        IN_PADDED_B, IN_PADDED_C, IN_PADDED_H, IN_PADDED_W)

    for bi in range(OUT_B):
        for ci in range(OUT_C):
            for hi in range(OUT_H):
                for wi in range(OUT_W):
                    pix_value = 0  # 输出中某批次、某通道的某像素值
                    for conv_ci in range(IN_C):
                        for conv_hi in range(KERNEL_H):
                            pix_value += IN_PADDED[bi][ci][hi+conv_hi][wi:wi+KERNEL_W].dot(KERNEL[ci][conv_ci][conv_hi])
                    
                    out[bi][ci][hi][wi] = pix_value

    return out


if __name__ == "__main__":
    # x = np.arange(2*3*4*5).reshape((2,3,4,5))
    # k = np.ones((4,3,3,3))
    # o = conv2d(x, k)
    # print(o.shape)


    import cv2

    img = cv2.imread(r'D:\workspace\deep_learning\data\g\mmexport1638545352010.jpg')
    k = np.array([
        [0.01, 0.1, 0.01],
        [0.01, 0.1, 0.01],
        [0.01, 0.1, 0.01],
    ])
    k = np.stack((k,k,k))
    k = np.stack((k,k,k))
    print('kernel shape: ',k.shape)  # (3, 3, 3, 3) cv2.resize(img, (10, 14))
    o = conv2d(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), k)
    # o = conv2d(np.expand_dims(np.transpose(cv2.resize(img, (108, 144)), (2, 0, 1)), axis=0), k)
    # o = conv2d(np.expand_dims(np.transpose(cv2.resize(img, (108*5, 144*5)), (2, 0, 1)), axis=0), k)

    cv2.imwrite('out.jpg', o.squeeze(axis=0).transpose(1,2,0))
    print('end')

    """

import numpy as np
from layers.conv2d import conv2d
x = np.arange(2*3*4*5).reshape((2,3,4,5))
k = np.ones((4,3,3,3))
o = conv2d(x, k)
print(o.shape)

    """
