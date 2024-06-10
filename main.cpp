#include "rbm.h"
#include <stdio.h>

int main(void){
    int v_num = 28*28;
    int h_num = 600;
    int epoch = 100;
    int sampling_num = 1000;
    int number = 5;
    RBM rbm(v_num, h_num);

    rbm.paramInput();

    printf("start read data\n");
    rbm.dataRead_MNIST(5000, number);
    printf("end read data\n");

    printf("start train\n");
    rbm.train_sampling(epoch, sampling_num);
    printf("end train\n");

    rbm.paramOutput();
    printf("out param.dat\n");


    return 0;
}