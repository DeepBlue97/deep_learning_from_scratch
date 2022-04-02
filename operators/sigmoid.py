import numpy as np
# import sympy as sy

def sigmoid(x: np.ndarray):
    """
    输出关于输入成非线性正相关，函数曲线像S，过点(0, 0.5)。
    输入：ndarray, (-inf, +inf)
    输出：ndarray,介于(0, 1)之间
    """
    return 1 / (1 + np.e**(-x))

if __name__ == "__main__":
    y = sigmoid(x = np.array([0]))
    print(y)
