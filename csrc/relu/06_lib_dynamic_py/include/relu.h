//
// Created by peter on 2022/4/12.
//

#ifndef DEEP_LEARNING_FROM_SCRATCH_RELU_H
#define DEEP_LEARNING_FROM_SCRATCH_RELU_H
class Relu
{
private:
    std::vector<bool> mask;

public:
    void showMask();
    std::vector<float> forward(const std::vector<float>& x);
};
#endif //DEEP_LEARNING_FROM_SCRATCH_RELU_H
