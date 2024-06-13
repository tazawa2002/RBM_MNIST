// rbm.h 制限ボルツマンマシンのヘッダーファイル
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdio.h>

using namespace std;

#ifndef RBM_H
#define RBM_H

class RBM {
    public:
        RBM(int v_num, int h_num);
        double energy_calc();
        double energy_v_calc();
        void update_v();
        void update_h();
        void sampling(int num);
        void dataGen(int num);
        void dataRead(int num);
        void train(int epoch);
        int traindatanum;
        vector< vector<int> > traindata;
        void paramOutput();
        void paramInput();
        void paramPrint();
        enum class TrainType {
            exact,
            sampling
        };
        enum class GradientType {
            nomal,
            momentum,
            nesterov,
            adagrad,
            rmsprop,
            adadelta,
            adam
        };
        enum class AnimeteType {
            anime, // アニメーションを実行する
            none // アニメーションを実行しない
        };
        TrainType train_type; // 訓練のタイプ
        int sampling_num; // サンプリングで使用するデータ数
        GradientType gradient_type; // 勾配法のタイプ
        AnimeteType animete_type; // アニメーションのタイプ
        void setTrainType(TrainType type);
        void setGradientType(GradientType type);
        void setAnimeteType(AnimeteType type);
        void setSamplingNum(int num);

        int vStates;
        int hStates;
        int totalStates;
        void p_distr_calc();
        void p_distr_v_calc();
        vector<double> p_distr;
        vector<double> p_distr_v;
        vector<int> histgram;
        vector<int> histgram_v;
    protected:
        vector<int> v;
        vector<int> h;
        vector<double> b;
        vector<double> c;
        vector< vector<double> > W;

        vector<double> Ev;
        vector<double> Eh;
        vector< vector<double> > Evh;
        vector<double> Ev_data;
        vector<double> Eh_data;
        vector< vector<double> > Evh_data;
        vector<double> gradient_b;
        vector<double> gradient_c;
        vector<vector<double> > gradient_w;

        double sig(double x);
        void exact_expectation();
        void sampling_expectation(int num);
        void data_expectation();
        void gradient_nomal(double learn_rate);
        void gradient_momentum(double learn_rate);
        void gradient_nesterov(double learn_rate);
        void gradient_adagrad(double learn_rate);
        void gradient_rmsprop(double learn_rate);
        void gradient_adadelta(double learn_rate);
        void gradient_adam(double learn_rate, int loop_time);
        void train_anime(int loop_time, int skip);

        void print();
        int state_num();
        int stateV();
        void setV(int num);
        void paramInit(int v_num, int h_num);
        void animeInit(int v_num, int h_num);
        double log_likelihood();

        // 乱数生成器のメンバ変数
        std::mt19937 gen;
        std::uniform_real_distribution<double> dis;
        double random_num();
};

#endif

