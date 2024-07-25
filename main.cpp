#include "rbm_mnist.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int v_num = 28*28;
    int h_num = 600;
    int epoch = 10;
    int sampling_num = 200;
    int number;
    RBM_MNIST rbm(v_num, h_num);
    rbm.setAnimeteType(RBM::AnimeteType::none);
    rbm.setTrainType(RBM::TrainType::sampling);
    rbm.setSamplingNum(sampling_num);
    rbm.setGradientType(RBM::GradientType::adam);

    if(argc > 1){
        number = atoi(argv[1]);
    }else{
        printf("select train number: ");
        scanf("%d", &number);
    }

    rbm.paramInput_MNIST(number);

    printf("start read data\n");
    rbm.dataRead_MNIST(60000, number);
    printf("end read data\n");

    printf("start train\n");
    rbm.trainMiniBatch(epoch, 200);
    printf("end train\n");

    rbm.paramOutput_MNIST(number);
    printf("out param.dat\n");


    return 0;
}