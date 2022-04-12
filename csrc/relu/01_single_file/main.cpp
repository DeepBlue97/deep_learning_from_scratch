//
// Created by 94504 on 2022/4/11.
//
#include <iostream>
#include <thread>
#include <chrono>
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

int main() {

//    std::vector<float> x(5);
    std::vector<float> x = {-1,3,-4,5,6};
    Relu relu = Relu();
    for(auto i : relu.forward(x)) {
        std::cout << i << std::endl;
    }
    relu.showMask();

//    auto t0 = std::chrono::steady_clock::now();
//    for (volatile int i = 0; i < 10000000; i++);
//    auto t1 = std::chrono::steady_clock::now();
//    auto dt = t1 - t0;
//    using double_ms = std::chrono::duration<double, std::milli>;
//    double ms = std::chrono::duration_cast<double_ms>(dt).count();
//    std::cout << "time elapsed: " << ms << " ms" << std::endl;
    return 0;
}
