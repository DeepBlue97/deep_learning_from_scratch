
# sigmoid

$f(x) = \frac{1}{1+e^{-x}}$

# softmax

$${\displaystyle \sigma (\mathbf {z} )_{j}={\frac {e^{z_{j}}}{\sum _{k=1}^{K}e^{z_{k}}}}} \quad for \ j = 1, …, K.$$

z为向量，元素个数为K。

# Jacobian matrix

When input and output were both vector, Jacobian was the gradient: partial derivative matrix(偏导数矩阵) 。


$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$


```
J00 ... J0n
.  .     .
.    .   .
.      . .
Jm0 ... Jmn
```

行为输出
f was output, which length is m.
x was input, which length is n.
