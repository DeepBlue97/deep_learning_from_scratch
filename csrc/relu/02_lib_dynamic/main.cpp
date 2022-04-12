//
// Created by 94504 on 2022/4/11.
//
#include <iostream>
#include <vector>

#include "relu.h"


void Relu::showMask() {
    std::cout << "Mask: " << std::endl;
    for(auto i:mask) {
        std::cout << i << std::endl;
    }
}

std::vector<float> Relu::forward(const std::vector<float>& x) {
    std::vector<float> y;

    for(auto i:x) {
        mask.push_back(i>0);
        if (i>0) y.push_back(i);
        else y.push_back(0);
    };
    return y;
};
