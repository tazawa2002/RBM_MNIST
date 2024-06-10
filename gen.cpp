#include "rbm.h"

int main(void){
    RBM rbm(28*28,600);
    int num = 200;
    int number = 5;

    rbm.dataRead_MNIST(1, number);
    rbm.paramInput();
    printf("end param read\n");
    rbm.dataGen_MNIST(num);
    printf("end data gen\n");

    return 0;
}