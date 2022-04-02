import numpy as np
import sympy as sy

def softmax(x: np.ndarray):
    """
    根据一个一维数组x，生成一个形状相同、值在0-1之间、总和为1的数组y。
    输出各元素可表示为概率。

    输入：x: array([1, 2, 3])
    输出：array([0.09003057, 0.24472847, 0.66524096])
    """
    x_exp = np.e ** x
    # sy.pprint(x_exp / x_exp.sum())
    return x_exp / x_exp.sum()

if __name__ == "__main__":
    softmax(x = np.array([1,2,3]))
