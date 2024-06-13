#include "rbm.h"

#ifndef RBM_MNIST_H
#define RBM_MNIST_H

class RBM_MNIST : public RBM {
public:
    RBM_MNIST(int v_num, int h_num) : RBM(v_num, h_num){};
    void dataGen_MNIST(int num, int number);
    void dataRead_MNIST(int num, int number);
    void paramOutput_MNIST(int number);
    void paramInput_MNIST(int number);
};

#endif