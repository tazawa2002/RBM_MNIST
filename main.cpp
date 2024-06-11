#include "rbm.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int v_num = 28*28;
    int h_num = 600;
    int epoch = 100;
    int sampling_num = 1000;
    int number;
    RBM rbm(v_num, h_num);

    if(argc > 1){
        number = atoi(argv[1]);
    }else{
        printf("select train number: ");
        scanf("%d", &number);
    }

    rbm.paramInput(number);

    printf("start read data\n");
    rbm.dataRead_MNIST(5000, number);
    printf("end read data\n");

    printf("start train\n");
    rbm.train_sampling(epoch, sampling_num);
    printf("end train\n");

    rbm.paramOutput(number);
    printf("out param.dat\n");


    return 0;
}