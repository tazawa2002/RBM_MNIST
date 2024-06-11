#include "rbm.h"

int main(int argc, char *argv[]){
    RBM rbm(28*28,600);
    int num = 200;
    int number;
    if(argc > 1){
        number = atoi(argv[1]);
    }else{
        printf("select gen number: ");
        scanf("%d", &number);
    }

    rbm.dataRead_MNIST(1, number);
    rbm.paramInput(number);
    printf("end param read\n");
    rbm.dataGen_MNIST(num, number);
    printf("end data gen\n");

    return 0;
}