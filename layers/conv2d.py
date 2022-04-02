import numpy as np


def conv2d(IN: np.ndarray, KERNEL: np.ndarray, PADDING: bool|int=0):
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

    # 定义一个中间张量IN_EXPAND，方便卷积
    IN_EXPAND_B = IN_B
    IN_EXPAND_C = IN_C
    IN_EXPAND_H = OUT_H * OUT_W  # 行数为out输出的高宽乘积
    IN_EXPAND_W = KERNEL_H * KERNEL_W  # 宽度为卷积核的高宽乘积
    IN_EXPAND = np.zeros((IN_EXPAND_B, IN_EXPAND_C, IN_EXPAND_H, IN_EXPAND_W))

    IN.reshape((IN_B, IN_C, -1, KERNEL_W))



    for bi in range(OUT_B):
        for ci in range(OUT_C):
            for hi in range(OUT_H):
                for wi in range(OUT_W):

                    out[bi][ci][hi][wi] = IN[bi][ci]


            pass

    # kernel_height = kernel.shape[2]
    # kernel_width = kernel.shape[3]

    # tensor_in_channel = tensor_in.shape[1]  # 输入通道数



    # for batch_index in range(tensor_out_batch):
    #     for channel_out_index in range(tensor_out_channel):
    #         tensor_2d_kernel = kernel[channel_out_index]
    #         for height_index in range(tensor_out_height):
    #             for width_index in range(tensor_out_width):
    #                 pix_channel_sum = 0  # 所有输入通道在某个像素位置的卷积之和
    #                 for channel_in_index in range(tensor_in_channel):
    #                     tensor_2d_in = tensor_in[
    #                         batch_index,
    #                         channel_in_index,
    #                         height_index:height_index+kernel_height,
    #                         width_index:width_index+kernel_width,
    #                     ]
    #                     pix_channel_sum += (tensor_2d_in * tensor_2d_kernel).sum()  # 不同输入通道之间累加
    #                 tensor_out[batch_index, channel_out_index, height_index, width_index] = pix_channel_sum
    # return tensor_out

