#include "rbm_mnist.h"

// データを生成する関数
void RBM_MNIST::dataGen_MNIST(int num, int number){
    int i, j, k;
    char filename[100];
    FILE *datafile;

    // バーンイン時間
    for(i=0;i<10000;i++){
        update_v();
        update_h();
    }

    // データ生成のループ
    for(k=0;k<num;k++){
        for(j=0;j<10;j++){
            update_v();
            update_h();
        }

        snprintf(filename, sizeof(filename), "./data/image%d-%d.dat", number, k);
        datafile = fopen(filename, "w");
        for(i=0;i<v.size();i++){
            fprintf(datafile, "%d ", v[i]);
            if(i%28==27) fprintf(datafile, "\n");
        }
        fclose(datafile);
        printf("\rmake %s", filename);
        fflush(stdout);
    }
    printf("\n");
}

void RBM_MNIST::dataRead_MNIST(int num, int number){
    int i, k, x;
    char filename[100];
    FILE *datafile;
    traindatanum = num;
    traindata.resize(traindatanum);
    for(i=0;i<traindatanum;i++){
        traindata[i].resize(v.size());
    }
    snprintf(filename, sizeof(filename), "./MNIST/train%d.dat", number);
    datafile = fopen(filename, "r");
    for(k=0;k<traindatanum;k++){
        for(i=0;i<v.size();i++){
            fscanf(datafile, "%d", &x);
            traindata[k][i] = x;
        }
    }
    fclose(datafile);
    for(i=0;i<v.size();i++){
        v[i] = traindata[0][i];
    }
}

void RBM_MNIST::paramOutput_MNIST(int number){
    int i, j;
    char filename[100];
    FILE *p;
    snprintf(filename, sizeof(filename), "./data/param%d.dat", number);
    p = fopen(filename, "w");
    if (p == NULL) {
        perror("Error opening p");
        return;
    }

    // 可視変数の数と隠れ変数の数を出力
    fprintf(p, "%ld %ld\n", v.size(), h.size());

    // パラメータbの出力
    for(i=0;i<v.size();i++){
        fprintf(p, "%lf ", b[i]);
    }
    fprintf(p, "\n");

    // パラメータcの出力
    for(j=0;j<h.size();j++){
        fprintf(p, "%lf ", c[j]);
    }
    fprintf(p, "\n");

    // パラメータWの出力
    for(i=0;i<v.size();i++){
        for(j=0;j<h.size();j++){
            fprintf(p, "%lf ", W[i][j]);
        }
        fprintf(p, "\n");
    }
    fclose(p);
}

void RBM_MNIST::paramInput_MNIST(int number){
    int i, j;
    int v_num, h_num;
    char filename[100];
    FILE *p;
    snprintf(filename, sizeof(filename), "./data/param%d.dat", number);
    p = fopen(filename, "r");
    if (p == NULL) {
        perror("Error opening p");
        return;
    }

    // 可視変数の数と隠れ変数の数を入力
    fscanf(p, "%d %d", &v_num, &h_num);
    paramInit(v_num, h_num);

    // パラメータbの入力
    for(i=0;i<v.size();i++){
        fscanf(p, "%lf", &b[i]);
    }

    // パラメータcの入力
    for(j=0;j<h.size();j++){
        fscanf(p, "%lf", &c[j]);
    }

    // パラメータWの入力
    for(i=0;i<v.size();i++){
        for(j=0;j<h.size();j++){
            fscanf(p, "%lf", &W[i][j]);
        }
    }
    fclose(p);
}

void RBM_MNIST::paramOutput_IMAGE(int number, int num){
    int i, j;
    char filename[100];
    FILE *datafile;

    // パラメータWの出力
    for(j=0;j<h.size() && j<num;j++){
        snprintf(filename, sizeof(filename), "./data/param%d-%d.dat", number, j);
        datafile = fopen(filename, "w");
        if (datafile == NULL) {
            perror("Error opening image param file");
            return;
        }
        for(i=0;i<v.size();i++){
            fprintf(datafile, "%lf ", W[i][j]);
            if(i%28==27) fprintf(datafile, "\n");
        }
        fclose(datafile);
        printf("\rmake %s", filename);
        fflush(stdout);
    }
}