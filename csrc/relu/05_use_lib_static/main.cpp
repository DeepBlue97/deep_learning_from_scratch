//
// Created by 94504 on 2022/4/11.
//
#include <iostream>
#include <vector>

#include "relu.h"


int main() {

    std::vector<float> x = {-1,3,-4,5,6};
    Relu relu = Relu();
    for(auto i : relu.forward(x)) {
        std::cout << i << std::endl;
    }
    relu.showMask();

    return 0;
}
