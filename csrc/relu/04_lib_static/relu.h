//
// Created by Administrator on 2022/4/12.
//

#ifndef CPPTEST_RELU_H
#define CPPTEST_RELU_H

#include <vector>

class Relu
{
private:
    std::vector<bool> mask;

public:
    void showMask();
    std::vector<float> forward(const std::vector<float>& x);
};
#endif //CPPTEST_RELU_H
