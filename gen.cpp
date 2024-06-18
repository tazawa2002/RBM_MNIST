#include "rbm_mnist.h"

int main(int argc, char *argv[]){
    RBM_MNIST rbm(28*28,600);
    int num = 200;
    int number;
    if(argc > 1){
        number = atoi(argv[1]);
    }else{
        printf("select gen number: ");
        scanf("%d", &number);
    }

    rbm.dataRead_MNIST(1, number);
    rbm.paramInput_MNIST(number);
    printf("end param read\n");
    rbm.dataGen_MNIST(num, number);
    printf("end data gen\n");
    rbm.paramOutput_IMAGE(number, 600);

    return 0;
}