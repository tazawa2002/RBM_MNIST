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

    rbm.paramInput_MNIST(10);

    rbm.dataRead_MNIST(100, number);
    for(number=0;number<100;number++){
        rbm.setV_random();
        rbm.dataGen_MNIST2(num, number);
    }
    

    rbm.paramOutput_IMAGE(number, 600);

    return 0;
}