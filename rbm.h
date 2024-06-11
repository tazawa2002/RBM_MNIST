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
        ~RBM();
        double energy_calc();
        double energy_v_calc();
        void update_v();
        void update_h();
        void sampling(int num);
        void dataGen(int num);
        void dataGen_MNIST(int num, int number);
        void dataRead(int num);
        void dataRead_MNIST(int num, int number);
        void train(int epoch);
        void train_sampling(int epoch, int num);
        int traindatanum;
        vector< vector<int> > traindata;
        void paramOutput(int number);
        void paramInput(int number);
        void paramPrint();

        int vStates;
        int hStates;
        int totalStates;
        void p_distr_calc();
        void p_distr_v_calc();
        vector<double> p_distr;
        vector<double> p_distr_v;
        vector<int> histgram;
        vector<int> histgram_v;
    private:
        vector<int> v;
        vector<int> h;
        vector< vector<double> > W;
        vector<double> b;
        vector<double> c;
        vector<double> Ev;
        vector<double> Eh;
        vector< vector<double> > Evh;
        vector<double> Ev_data;
        vector<double> Eh_data;
        vector< vector<double> > Evh_data;
        double sig(double x);
        void exact_expectation();
        void sampling_expectation(int num);
        void data_expectation();
        void train_anime(int loop_time, int skip);

        void print();
        int state_num();
        int stateV();
        void setV(int num);
        void paramInit(int v_num, int h_num);
        double log_likelihood();

        // 乱数生成器のメンバ変数
        std::mt19937 gen;
        std::uniform_real_distribution<double> dis;
        double random_num();
};

#endif

