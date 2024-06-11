#include "rbm.h"

// コンストラクタ
RBM::RBM(int v_num, int h_num){
    // 変数の用意
    paramInit(v_num, h_num);

    std::random_device rd;
    gen = std::mt19937(rd()); // 乱数生成器の初期化
    dis = std::uniform_real_distribution<double>(0.0, 1.0); // 0.0から1.0の範囲で乱数を生成

    // 変数の初期化
    for(int i = 0; i < this->v.size(); i++){
        for(int j = 0; j < this->h.size(); j++){
            this->W[i][j] = 2*random_num() - 1;
        }
    }
    for(int i = 0; i < this->v.size(); i++){
        this->b[i] = 2*random_num() - 1;
    }
    for(int i = 0; i < this->h.size(); i++){
        this->c[i] = 2*random_num() - 1;
    }
}

// デストラクタ
RBM::~RBM(){
    int i;
    i = 0;
}

// 0.0から1.0の範囲で乱数を生成
double RBM::random_num(){
    return dis(gen);
}

// エネルギーを計算する関数
double RBM::energy_calc(){
    int i,j;
    double energy = 0;
    for(i=0;i<this->v.size();i++){
        for(j=0;j<this->h.size();j++){
            energy -= W[i][j]*v[i]*h[j];
        }
    }
    for(i=0;i<this->v.size();i++){
        energy -= b[i]*v[i];
    }
    for(j=0;j<this->h.size();j++){
        energy -= c[j]*h[j];
    }
    return energy;
}

double RBM::energy_v_calc(){
    int i,j;
    double lambda;
    double energy = 0;
    for(i=0;i<this->v.size();i++){
        energy -= b[i]*v[i];
    }
    for(j=0;j<this->h.size();j++){
        lambda = c[j];
        for(i=0;i<this->v.size();i++){
            lambda += W[i][j]*v[i];
        }

        // 指数関数の発散を防ぐ処理を追加
        // 1 << exp(lambda)のとき log(1+exp(lambda)) ≒ lambda
        if(lambda > 20){
            energy -= lambda;
        }else{
            energy -= log(1+exp(lambda));
        }
    }
    return energy;
}

// 確率を計算
void RBM::p_distr_calc(){
    int i,j,k;
    double Z = 0;

    // すべての状態の確率を求める
    for(k=0;k<totalStates;k++){
        // 状態を設定
        for(i=0;i<v.size();i++){
            v[i] = (k >> i)&1;
        }
        for(j=0;j<h.size();j++){
            h[j] = (k >> (j+v.size()))&1;
        }
        p_distr[k] = exp(-energy_calc());
    }
    for(i=0;i<this->totalStates;i++){
        Z += p_distr[i];
    }
    for(i=0;i<totalStates;i++){
        p_distr[i] = p_distr[i] / Z;
    }
}

// 確率を計算
void RBM::p_distr_v_calc(){
    int i, k;
    double Z = 0;

    // 配列の初期化
    for(k=0;k<vStates;k++){
        for(i=0;i<v.size();i++){
            v[i] = (k >> i) & 1;
        }
        p_distr_v[k] = exp(-energy_v_calc());
    }

    for(k=0;k<this->vStates;k++){
        Z += p_distr_v[k];
    }

    for(k=0;k<vStates;k++){
        p_distr_v[k] = p_distr_v[k] / Z;
    }
}

// 可視層の状態を更新
void RBM::update_v(){
    int i, j;
    double lambda;
    double p, u;
    for(i=0;i<v.size();i++){
        lambda = b[i];
        for(j=0;j<h.size();j++){
            lambda += W[i][j]*h[j];
        }
        p = sig(lambda);
        u = random_num();
        if(p >= u){
            v[i] = 1;
        }else{
            v[i] = 0;
        }
    }
}

// 隠れ層の状態を更新
void RBM::update_h(){
    int i, j;
    double lambda;
    double p, u;
    for(j=0;j<h.size();j++){
        lambda = c[j];
        for(i=0;i<v.size();i++){
            lambda += W[i][j]*v[i];
        }
        p = sig(lambda);
        u = random_num();
        if(p >= u){
            h[j] = 1;
        }else{
            h[j] = 0;
        }
    }
}

// サンプリング
void RBM::sampling(int num){
    int i, j;
    for(i=0;i<vStates;i++){
        histgram_v[i] = 0;
    }

    for(i=0;i<1000;i++){
        update_v();
        update_h();
    }

    for(i=0;i<num;i++){
        for(j=0;j<10;j++){
            update_v();
            update_h();
        }
        histgram_v[stateV()] += 1;
    }
}

// シグモイド関数
double RBM::sig(double x){
    double exp_val;
    if(x>0){
        return 1 / ( 1 + exp(-x) );
    }else{
        exp_val = exp(x);
        return exp_val / ( 1 + exp_val);
    }
}

void RBM::print(){
    int i;
    printf("data:");
    for(i=0;i<v.size();i++){
        printf("%d ", v[i]);
    }
    for(i=0;i<h.size();i++){
        printf("%d ", h[i]);
    }
    printf("\n");
}

int RBM::state_num(){
    int i;
    int state = 0;
    for(i=0;i<v.size();i++){
        state += v[i] * pow(2,i);
    }
    for(i=0;i<h.size();i++){
        state += h[i] * pow(2,i+v.size());
    }
    return state;
}

// データを生成する関数
void RBM::dataGen(int num){
    int i, j, k;
    FILE *datafile;
    datafile = fopen("./data/data.dat", "w");
    if(datafile == NULL) exit(1);

    // バーンイン時間
    for(i=0;i<1000;i++){
        update_v();
        update_h();
    }

    printf("start data gen");
    // データ生成のループ
    for(k=0;k<num;k++){
        for(j=0;j<1;j++){
            update_v();
            update_h();
        }
        for(i=0;i<v.size();i++){
            fprintf(datafile, "%d ", v[i]);
        }
        fprintf(datafile, "\n");
    }
    fclose(datafile);
    printf("end data gen");
}

// データを生成する関数
void RBM::dataGen_MNIST(int num, int number){
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
    }
    fclose(datafile);
}

// データを読み込む関数
void RBM::dataRead(int num){
    int i, k, x;
    FILE *datafile;
    traindatanum = num;
    traindata.resize(traindatanum);
    for(i=0;i<traindatanum;i++){
        traindata[i].resize(v.size());
    }
    datafile = fopen("./MNIST/train0.dat", "r");
    for(k=0;k<traindatanum;k++){
        for(i=0;i<v.size();i++){
            fscanf(datafile, "%d", &x);
            traindata[k][i] = x;
        }
    }
    fclose(datafile);
}

void RBM::dataRead_MNIST(int num, int number){
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

void RBM::train(int epoch){
    int i,j,k;
    int loop_time = 0;
    double learn_rate = 0.01;
    double lambda;
    double v_ave_model;
    double v_ave_data;
    double h_ave_model;
    double h_ave_data;
    double vh_ave_model;
    double vh_ave_data;
    double gradient = 10;
    vector<double> gradient_b;
    vector<double> gradient_c;
    vector<vector<double> > gradient_w;
    gradient_b.resize(v.size());
    gradient_c.resize(h.size());
    gradient_w.resize(v.size());
    for(i=0;i<v.size();i++){
        gradient_w[i].resize(h.size());
    }

    // 対数尤度関数の出力ファイルの準備
    FILE *p;
    p = fopen("./data/log_likelihood.dat", "w");
    if(p == NULL){
        perror("Error opening log_likelihood.dat");
        exit(1);
    }

    std::cout << "\e[?25l"; // カーソルを非表示
    while(loop_time < epoch){

        exact_expectation(); // 期待値計算
        train_anime(loop_time, 10); // アニメーション用のファイル出力
        fprintf(p, "%d %lf\n", loop_time, log_likelihood()); // 

        // パラメータの更新
        for(i=0;i<v.size();i++){

            // v_iのデータ平均を求める処理
            v_ave_data = 0;
            for(k=0;k<traindatanum;k++){
                v_ave_data += traindata[k][i];
            }
            v_ave_data /= traindatanum;

            gradient_b[i] = v_ave_data - Ev[i];
        }

        for(j=0;j<h.size();j++){
            // h_iのデータ平均を求める処理
            h_ave_data = 0;
            for(k=0;k<traindatanum;k++){
                // lambdaを計算する
                lambda = c[j];
                for(i=0;i<v.size();i++){
                    lambda += W[i][j]*traindata[k][i];
                }
                h_ave_data += sig(lambda);
            }
            h_ave_data /= traindatanum;

            // h_jのモデル平均
            gradient_c[j] = h_ave_data - Eh[j];
        }

        for(i=0;i<v.size();i++){
            for(j=0;j<h.size();j++){
                // vhのデータ平均を求める処理
                vh_ave_data = 0;
                for(k=0;k<traindatanum;k++){
                    lambda = c[j];
                    for(int l=0;l<v.size();l++){
                        lambda += W[l][j]*traindata[k][l];
                    }
                    vh_ave_data += traindata[k][i]*sig(lambda);
                }
                vh_ave_data /= traindatanum;

                gradient_w[i][j] = vh_ave_data - Evh[i][j];
            }
        }

        // パラメータの更新と勾配の計算
        gradient = 0;
        for(i=0;i<v.size();i++){
            gradient += gradient_b[i]*gradient_b[i];
            b[i] += learn_rate*gradient_b[i];
        }
        for(j=0;j<h.size();j++){
            gradient += gradient_c[j]*gradient_c[j];
            c[j] += learn_rate*gradient_c[j];
        }
        for(i=0;i<v.size();i++){
            for(j=0;j<h.size();j++){
                gradient += gradient_w[i][j]*gradient_w[i][j];
                W[i][j] += learn_rate*gradient_w[i][j];
            }
        }
        gradient = sqrt(gradient);
        std::cout << "\r" << loop_time << ": " << gradient;
        if(loop_time%100 == 0) fflush(stdout);
        loop_time++;
    }
    std::cout << "\e[?25h" << endl; // カーソルの再表示
    exact_expectation(); // 期待値計算
    train_anime(loop_time, 10); // アニメーション用のファイル出力
    fprintf(p, "%d %lf\n", loop_time, log_likelihood()); 
    fflush(p);
    fclose(p);
}

void RBM::train_sampling(int epoch,int num){
    int i,j,k;
    int loop_time = 0;
    double learn_rate = 0.01;
    double gradient = 10;
    vector<double> gradient_b;
    vector<double> gradient_c;
    vector<vector<double> > gradient_w;
    gradient_b.resize(v.size());
    gradient_c.resize(h.size());
    gradient_w.resize(v.size());
    for(i=0;i<v.size();i++){
        gradient_w[i].resize(h.size());
    }

    FILE *p;
    p = fopen("./data/gradient.dat", "w");

    printf("strat train data ave\n");
    data_expectation();
    printf("end train data ave\n");
    for(i=0;i<v.size();i++){
        v[i] = traindata[0][i];
    }

    std::cout << "\e[?25l"; // カーソルを非表示
    while(loop_time < epoch){
        sampling_expectation(num);

        // パラメータの更新
        gradient = 0;
        for(i=0;i<v.size();i++){
            gradient_b[i] = Ev_data[i] - Ev[i];
            gradient += gradient_b[i]*gradient_b[i];
            b[i] += learn_rate*gradient_b[i];
        }
        for(j=0;j<h.size();j++){
            gradient_c[j] = Eh_data[j] - Eh[j];
            gradient += gradient_c[j]*gradient_c[j];
            c[j] += learn_rate*gradient_c[j];
        }
        for(j=0;j<h.size();j++){
            for(i=0;i<v.size();i++){
                gradient_w[i][j] = Evh_data[i][j] - Evh[i][j];
                gradient += gradient_w[i][j]*gradient_w[i][j];
                W[i][j] += learn_rate*gradient_w[i][j];
            }
        }
        gradient = sqrt(gradient);
        loop_time++;

        std::cout << "\r" << loop_time << ": " << gradient << std::endl;
        fflush(stdout);
        fprintf(p, "%d %lf\n", loop_time, gradient);
    }
    sampling_expectation(num);
    fflush(p);
    fclose(p);
    std::cout << "\e[?25h" << endl; // カーソルの再表示
}

void RBM::train_anime(int loop_time, int skip){
    // アニメーション用のファイルを出力
    int i;
    char filename[100];
    FILE *p;
    p_distr_v_calc();
    if(loop_time%skip == 0){
        snprintf(filename, sizeof(filename), "./data/learn-v-%03d.dat", loop_time/skip);
        p = fopen(filename, "w");
        if (p != NULL) {
            for(i=0;i<vStates;i++){
                fprintf(p, "%d %lf\n", i, p_distr_v[i]);
            }
            fclose(p);
        } else {
            perror("Error opening p");
        }
    }
}

void RBM::exact_expectation(){
    int i, j, k;
    double lambda;
    p_distr_v_calc();
    for(i=0;i<v.size();i++){
        // v_iのモデル平均
        Ev[i] = 0;
        for(k=0;k<vStates;k++){
            Ev[i] += p_distr_v[k]*((k>>i)&1);
        }
    }
    for(j=0;j<h.size();j++){
        // h_jのモデル平均
        Eh[j] = 0;
        for(k=0;k<vStates;k++){
            lambda = c[j];
            for(i=0;i<v.size();i++){
                lambda += W[i][j]*((k>>i)&1);
            }
            Eh[j] += p_distr_v[k]*sig(lambda);
        }
    }
    for(i=0;i<v.size();i++){
        for(j=0;j<h.size();j++){
            // vhのモデル平均
            Evh[i][j] = 0;
            for(k=0;k<vStates;k++){
                lambda = c[j];
                for(int l=0;l<v.size();l++){
                    lambda += W[l][j]*((k>>l)&1);
                }
                Evh[i][j] += ((k>>i)&1)*sig(lambda)*p_distr_v[k];
            }
        }
    }
}

void RBM::sampling_expectation(int num){
    int i, j, k, l;
    
    for(i=0;i<v.size();i++){
        Ev[i] = 0;
    }
    for(j=0;j<h.size();j++){
        Eh[j] = 0;
    }
    for(i=0;i<v.size();i++){
        for(j=0;j<h.size();j++){
            Evh[i][j] = 0;
        }
    }

    for(k=0;k<0;k++){
        update_v();
        update_h();
    }

    printf("start sampling\n");fflush(stdout);
    for(k=0;k<num;k++){
        for(l=0;l<1;l++) {
            update_v();
            update_h();
        }
        
        // 期待値を足す
        for(i=0;i<v.size();i++){
            Ev[i] += v[i];
        }
        for(j=0;j<h.size();j++){
            Eh[j] += h[j];
        }
        for(i=0;i<v.size();i++){
            for(j=0;j<h.size();j++){
                Evh[i][j] += v[i]*h[j];
            }
        }
    }
    printf("end sampling\n");fflush(stdout);

    // データ数で割る
    for(i=0;i<v.size();i++){
        Ev[i] /= num;
    }
    for(j=0;j<h.size();j++){
        Eh[j] /= num;
    }
    for(i=0;i<v.size();i++){
        for(j=0;j<h.size();j++){
            Evh[i][j] /= num;
        }
    }
}

void RBM::data_expectation(){
    int i, j, k;
    double lambda;
    vector<double> sig_lambda;
    sig_lambda.resize(traindatanum);

    for(i=0;i<v.size();i++){
        // v_iのデータ平均を求める処理
        Ev_data[i] = 0;
        for(k=0;k<traindatanum;k++){
            Ev_data[i] += traindata[k][i];
        }
        Ev_data[i] /= traindatanum;
    }
    for(j=0;j<h.size();j++){
        // h_iのデータ平均を求める処理
        Eh_data[j] = 0;
        for(k=0;k<traindatanum;k++){
            // lambdaを計算する
            lambda = c[j];
            for(i=0;i<v.size();i++){
                lambda += W[i][j]*traindata[k][i];
            }
            sig_lambda[k] = sig(lambda);
            Eh_data[j] += sig_lambda[k];
        }
        Eh_data[j] /= traindatanum;

        for(i=0;i<v.size();i++){
            // vhのデータ平均を求める処理
            Evh_data[i][j] = 0;
            for(k=0;k<traindatanum;k++){
                Evh_data[i][j] += traindata[k][i]*sig_lambda[k];
            }
            Evh_data[i][j] /= traindatanum;
        }
    }
}

// 可視変数の状態を2進数で返す関数
int RBM::stateV(){
    int i;
    int num = 0;
    for(i=0;i<v.size();i++){
        num += pow(2,i)*v[i];
    }
    return num;
}

void RBM::setV(int num){
    int i;
    for(i=0;i<v.size();i++){
        v[i] = (num >> i) & 1;
    }
}

void RBM::paramOutput(int number){
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

void RBM::paramInput(int number){
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

void RBM::paramPrint(){
    int i, j;
    for(i=0;i<v.size();i++){
        printf("%lf ", b[i]);
    }
    printf("\n");

    for(j=0;j<h.size();j++){
        printf("%lf ", c[j]);
    }
    printf("\n");

    for(i=0;i<v.size();i++){
        for(j=0;j<h.size();j++){
            printf("%lf ", W[i][j]);
        }
        printf("\n");
    }
}

void RBM::paramInit(int v_num, int h_num){
    // 変数の用意
    this->v.resize(v_num);
    this->h.resize(h_num);
    this->W.resize(v_num);
    for(int i=0;i<v_num;i++){
        this->W[i].resize(h_num);
    }
    this->b.resize(v_num);
    this->c.resize(h_num);
    this->Ev.resize(v_num);
    this->Eh.resize(h_num);
    this->Evh.resize(v_num);
    for(int i=0;i<v_num;i++){
        this->Evh[i].resize(h_num);
    }
    Ev_data.resize(v_num);
    Eh_data.resize(h_num);
    Evh_data.resize(v_num);
    for(int i=0;i<v_num;i++){
        Evh_data[i].resize(h_num);
    } 

    // 厳密計算に必要な変数
    // this->vStates = pow(2,v_num);
    // this->hStates = pow(2,h_num);
    // this->totalStates = this->vStates*this->hStates;
    // this->p_distr_v.resize(this->vStates);
    // this->histgram_v.resize(this->vStates);
}

double RBM::log_likelihood(){
    int i, mu;
    int state;
    double lambda;
    double log_likelihood = 0;
    for(mu=0;mu<traindatanum;mu++){
        state = 0;
        for(i=0;i<v.size();i++){
            state += traindata[mu][i]*pow(2,i);
        }
        log_likelihood += log(p_distr_v[state]);
    }
    return log_likelihood / traindatanum;
}