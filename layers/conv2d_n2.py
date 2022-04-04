import numpy as np

"""
该算法我愿称之为"稀疏卷积核"方法，当图片较大时，效率极其低下！算法复杂度为 n^2

-------------------------------------------------------
想到一个修复的方法：对输出张量中没用的元素进行过滤

I've got a great idea to fix this problem: 
    filter out the useless elements of the out-tensor.

-------------------------------------------------------
这是一个失败的思路： 卷积核多个通道合并， 一次性卷积, 
这会导致在不该卷积的时候进行卷积（如卷积核最右侧的元素不可能和图像最左侧的元素进行乘积）。

This is a unsuccessful algorithm: 
    kernel flattened in channel dimention, 
    so it can convolve only one for multiple channels.

This will lead to convolve even when need't convolve, 
such as the most right elements of kernel time by the most left elements of image.
"""

def conv2d(IN: np.ndarray, KERNEL: np.ndarray, PADDING: bool | int = 0):
    """
    input:
        IN: (IN_B, IN_C, IN_H, IN_W)
        KERNEL: (KERNEL_B, KERNEL_C=IN_C, KERNEL_H, KERNEL_W)

    output: 
        out: (OUT_B=IN_B, OUT_C=KERNEL_B, OUT_H, OUT_W)

    """
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

    # # 定义一个中间张量IN_EXPAND，方便卷积
    # IN_EXPAND_B = IN_B
    # IN_EXPAND_C = IN_C
    # IN_EXPAND_H = OUT_H * OUT_W  # 行数为out输出的高宽乘积
    # IN_EXPAND_W = KERNEL_H * KERNEL_W  # 宽度为卷积核的高宽乘积
    # IN_EXPAND = np.zeros((IN_EXPAND_B, IN_EXPAND_C, IN_EXPAND_H, IN_EXPAND_W))

    # IN.reshape((IN_B, IN_C, -1, KERNEL_W))

    # 输入张量PAD之后的张量，真正待卷积的张量
    IN_PADDED_B = IN_B
    IN_PADDED_C = IN_C
    IN_PADDED_H = IN_H + PADDING
    IN_PADDED_W = IN_W + PADDING

    # 自动设置PADDING量
    if PADDING is True:
        PADDING = int((KERNEL_W - 1) / 2)
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
    # 仅保留批次维度，用于后续一维卷积
    IN_PADDED_FLATTEN = IN_PADDED.reshape((IN_PADDED_B, -1))

    KERNEL_EXPAND = np.concatenate(
        (
            np.concatenate(
                (KERNEL, np.zeros((KERNEL_B, KERNEL_C, KERNEL_H, IN_PADDED_W-KERNEL_W))),
                axis=-1  # 列维度扩张
            ),
            np.zeros((KERNEL_B, KERNEL_C, IN_PADDED_H-KERNEL_H, IN_PADDED_W))
        ),
        axis=-2  # 行维度扩张
    )

    KERNEL_EXPAND = KERNEL_EXPAND.reshape((KERNEL_B, -1))[
        :, # 批次维度全要
        # 去掉卷积核多余的尾巴，使得可以卷积直到图像的最后一个像素
        0:-((IN_PADDED_H-KERNEL_H)*IN_PADDED_W+(IN_PADDED_W-KERNEL_W))
        # 0:(IN_PADDED_C-1)*(IN_PADDED_H*IN_PADDED_W)  # 卷积核通道跨度
        # + (KERNEL_H*KERNEL_W)  # 卷积核单通道的长度
        # + (KERNEL_H-1)*(IN_PADDED_W-KERNEL_W)  # 卷积核行间填充
    ]

    # # 定义一个中间张量KERNEL_EXPAND，可以做到多通道同时卷积
    # KERNEL_EXPAND_B = KERNEL_B
    # KERNEL_EXPAND = np.zeros(
    #     (IN_PADDED_C-1)*(IN_PADDED_H*IN_PADDED_W)  # 卷积核通道跨度
    #     + (KERNEL_H*KERNEL_W)  # 卷积核单通道的长度
    #     + (KERNEL_H-1)*(IN_PADDED_W-KERNEL_W)  # 卷积核行间填充
    # ).reshape((KERNEL_EXPAND_B, -1))

    # 由于卷积核展平后会导致不该卷积的时候卷积，如卷积核最右侧的元素不可能和图像最左侧的元素进行乘积
    EXEC_CYCLE = IN_PADDED_W - KERNEL_W + 1  # 连续执行卷积的次数
    SKIP_CYCLE = KERNEL_W - 1  # 连续continue跳过卷积的次数
    for bi in range(OUT_B):  # 输出张量批次循环
        for ci in range(OUT_C):  # 输出张量通道循环
            print('一个通道开始！')
            # one piece of channel of out-tensor
            out_channel = np.convolve(
                IN_PADDED_FLATTEN[bi], KERNEL_EXPAND[ci], mode='valid'
            )
            # (由于存在冗余元素)过滤掉多余元素后的张量 out-tensor which filtered out useless elements
            out_channel_filtered = np.zeros(OUT_H * OUT_W)
            count_valid = 0  # count the number of elements input to the out_channel_filtered
            for i in range(out_channel.size):
                if (i % (EXEC_CYCLE+SKIP_CYCLE)) < EXEC_CYCLE:
                    out_channel_filtered[count_valid] = out_channel[i]
                    count_valid += 1

            # out的channel index对应的是kernel的batch index
            out[bi][ci] = out_channel_filtered.reshape((OUT_H, OUT_W))

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
    print(k.shape)  # (3, 3, 3, 3) cv2.resize(img, (10, 14))
    # o = conv2d(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), k)
    # o = conv2d(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0), k)
    # o = conv2d(np.expand_dims(np.transpose(cv2.resize(img, (108, 144)), (2, 0, 1)), axis=0), k)
    o = conv2d(np.expand_dims(np.transpose(cv2.resize(img, (108*5, 144*5)), (2, 0, 1)), axis=0), k)

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
