#include "rbm.h"
#include <stdio.h>

int main(int argc, char *argv[]){
    int v_num = 28*28;
    int h_num = 600;
    int epoch = 1000;
    int sampling_num = 100;
    int number;
    RBM rbm(v_num, h_num);
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

    rbm.paramInput(number);

    printf("start read data\n");
    rbm.dataRead_MNIST(5000, number);
    printf("end read data\n");

    printf("start train\n");
    rbm.train(epoch);
    printf("end train\n");

    rbm.paramOutput(number);
    printf("out param.dat\n");


    return 0;
}