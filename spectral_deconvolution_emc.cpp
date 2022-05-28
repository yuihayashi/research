#include<bits/stdc++.h>
using namespace std;


random_device rd;
default_random_engine eng(rd());
random_device seed;
mt19937 engine(seed());


vector<vector<double>> generate_data() {
	normal_distribution<double> gauss_noise(0., 0.1);
    vector<double> a_nums{0.587, 1.522, 1.183};
    vector<double> mu_nums{1.210, 1.455, 1.703};
    vector<double> b_nums{95.689, 146.837, 164.469};
    vector<double> x_vec(301);
    vector<double> y_vec(301);
    vector<vector<double>> data_vec(2, vector<double>(301, 0.));

    double x = 0.;
    double y = 0.;

    for (int i=0; i<301; ++i) {
        x = (i + 1) * 0.01;
        y = a_nums[0] * exp(-b_nums[0] / 2. * pow((x - mu_nums[0]), 2)) + \
            a_nums[1] * exp(-b_nums[1] / 2 * pow((x - mu_nums[1]), 2)) + \
            a_nums[2] * exp(-b_nums[2] / 2. * pow((x - mu_nums[2]), 2)) + \
            gauss_noise(eng);
        data_vec[0][i] = x;
        data_vec[1][i] = y;
    }
    return data_vec;
}

// 事前確率の定義 done
double a_density(double ak) {
    if (ak < 0) {
        return 0.;
    }
    else {
        return pow(ak, 4) * exp(-5 * ak);
    }
}

double mu_density(double muk) {
    return exp(-5. / 2 * pow((muk - 1.5), 2));
}

double b_density(double bk) {
    if (bk < 0) {
        return 0.;
    }
    else {
        return pow(bk, 4) * exp(-0.04 * bk);
    }
}

// beta list の定義
vector<double> make_beta_vec(int replica_size) {
    vector<double> beta_vec(replica_size);
    for (int l=1; l<replica_size+1; ++l) {
        if(l==1) {
            beta_vec[l-1] = 0.;
        }
        else {
            beta_vec[l-1] = pow(1.5, (double)(l-replica_size));
        }
    }
    return beta_vec;
}

// エネルギー関数
double energy_func(int K, vector<vector<double>> theta_l, vector<double> x_data, vector<double> y_data) {
    vector<double> a_vec = theta_l[0];
    vector<double> mu_vec = theta_l[1];
    vector<double> b_vec = theta_l[2];
    vector<double> f_vec(x_data.size(), 0.); 
    double f_k = 0.;
    double E = 0.;

    for (int i=0; i<x_data.size(); ++i) {
        f_k = 0.;
        for (int k=0; k<K; ++k) {
            f_k += a_vec[k] * exp((-b_vec[k] / 2) * pow((x_data[i] - mu_vec[k]), 2));
        }
        f_vec[i] = f_k;
    }
    for (int i=0; i<x_data.size(); ++i) {
        E += pow((y_data[i] - f_vec[i]) , 2) / (2. * x_data.size());
    }
    return E;
}

// 永田の関係式
double nagata_step(vector<vector<double>> step_Cd, vector<double> x_data, double beta, int j) {
    int n = x_data.size();
    if (n * beta < 1.) {
        return step_Cd[j][0];
    }
    else {
        return step_Cd[j][0] / pow((n * beta), step_Cd[j][1]);
    }
}

// レプリカの交換を行う
vector<vector<vector<vector<double>>>> exchange_replica(vector<vector<vector<vector<double>>>> theta,\
                                                        vector<double> x_data, vector<double> y_data, \
                                                        vector<double> beta_vec, int replica_size, int K, vector<double> &exchange_count) {
    for (int l=0; l<replica_size-1; ++l) {
        vector<vector<double>> theta1(3, vector<double>(K));
        vector<vector<double>> theta2(3, vector<double>(K));

        int ex_idx1 = l;
        int ex_idx2 = l + 1;

        theta1 = theta[0][ex_idx1];
        theta2 = theta[0][ex_idx2];

        int n = x_data.size();
        double v = exp(n / 0.01 * (beta_vec[ex_idx2] - beta_vec[ex_idx1]) * \
                        (energy_func(K, theta2, x_data, y_data) - energy_func(K, theta1, x_data, y_data)));
        double u = min(1., v);
        uniform_real_distribution<double> distr2(0., 1.);

        if (distr2(eng) < v) {
            ++exchange_count[l];
            swap(theta[0][ex_idx1], theta[0][ex_idx2]);
        }
    }
    return theta;
}

//　真分布 
double q_probability_distribution(vector<vector<double>> theta_l, vector<double> x_data, vector<double> y_data, int target_k, double beta, int K) {
    int n = x_data.size();
    double q = exp(-n * beta * 100. * energy_func(K, theta_l, x_data, y_data)) * \
              a_density(theta_l[0][target_k]) * mu_density(theta_l[1][target_k]) * b_density(theta_l[2][target_k]);
    return q;
}

// // 確率の比
// double q_probability_ratio(vector<vector<double>> theta_1, vector<vector<double>> theta_2, \
//                            vector<double> x_data, vector<double> y_data, int target_k, double beta, int K) {
//     cout << signbit(-(x_data.size() / 0.01)) << endl;
//     double q_ratio = exp(-(x_data.size() / 0.01) * beta * (energy_func(K, theta_2, x_data, y_data) - energy_func(K, theta_1, x_data, y_data))) *  \
//            a_density(theta_2[0][target_k])/a_density(theta_1[0][target_k]) * \
//            mu_density(theta_2[1][target_k])/mu_density(theta_1[1][target_k]) * \
//            b_density(theta_2[2][target_k])/b_density(theta_1[2][target_k]);
//     return q_ratio;
// }

// メトロポリス法
vector<vector<double>> _propose_theta_for_params(vector<vector<double>> _params, vector<vector<double>> step_Cd, \
                                                 vector<double> x_data, vector<double> y_data, double beta, \
                                                 int target_rep_idx, int j, int k, int K, vector<vector<double>> &acceptance_count) {
    double base_params = _params[j][k];
    vector<vector<double>> proposed_params = _params;

    uniform_real_distribution<double> distr1(-1., 1.);
    proposed_params[j][k] = base_params + nagata_step(step_Cd, x_data, beta, j) * distr1(eng);

    double _q = q_probability_distribution(proposed_params, x_data, y_data, k, beta, K);
    double q = q_probability_distribution(_params, x_data, y_data, k, beta, K);
    double q_ratio = _q / q;
    uniform_real_distribution<double> distr2(0., 1.);

    if (distr2(eng) < q_ratio) { 
        ++acceptance_count[target_rep_idx][j];
        return proposed_params;
    }
    else {
        return _params;
    }
}

//　新しい変数を返す
vector<vector<vector<vector<double>>>> next_with_metropolis(vector<vector<vector<vector<double>>>> params_vec, vector<vector<double>> step_Cd, \
                                                            vector<double> x_data, vector<double> y_data, int target_rep_idx, double beta, int K, vector<vector<double>> &acceptance_count) {
    vector<vector<double>> params = params_vec[0][target_rep_idx];  // (3, K_nums)
    double check;
    for (int j=0; j<3; ++j) {
        for (int k=0; k<K; ++k) {
            params = _propose_theta_for_params(params, step_Cd, x_data, y_data, beta, target_rep_idx, j, k, K, acceptance_count);
        }
    }
    params_vec[0][target_rep_idx] = params;
    return params_vec;
}

// 周辺尤度を計算する
double culc_z(int K, int iter_nums, int burn_in, int replica_size, \
              vector<vector<vector<vector<double>>>> all_theta, vector<double> x_data, vector<double> y_data, vector<double> beta_vec) {
    int n = x_data.size();
    double E_z = 1.;
    double z;

    for (int l=0; l<replica_size-1; ++l) {
        z = 0.;
        for (int i=0; i<(iter_nums-burn_in); ++i) {
            z += exp(-n * 100. * (beta_vec[l+1] - beta_vec[l]) * energy_func(K, all_theta[i][l], x_data, y_data)) / (double)(iter_nums-burn_in);
        }
        E_z *= z;
    }
    return E_z;
}


int main() {
    // 正規分布
	normal_distribution<double> gauss(1.5, sqrt(1/5.));  // mu
    // ガンマ分布
    gamma_distribution<double> gamma1(5., 1/5.);  // a
    gamma_distribution<double> gamma2(5., 1/0.04); // b

    // 定数
    const int iter_nums = 4 * pow(10, 4);
    const int burn_in = 2 * pow(10, 4);
    const int replica_size = 24;
    // ハイパーパラメータの宣言
    int K_nums = 3;
    vector<vector<double>> step_Cd{{1.5, 0.5}, {0.8, 0.8}, {150., 0.3}};

    vector<vector<double>> acceptance_count(replica_size, vector<double>(3, 0));
    vector<double> exchange_count(replica_size);
    vector<vector<vector<vector<double>>>> stored_theta((iter_nums - burn_in), vector<vector<vector<double>>>(replica_size, \
                                                         vector<vector<double>>(3, vector<double>(K_nums))));

     // thetaのa, mu, bの初期状態を決める
    vector<vector<vector<vector<double>>>> theta(1, vector<vector<vector<double>>>(replica_size, vector<vector<double>>(3, vector<double>(K_nums))));
    for (int l=0; l<replica_size; ++l) {
        for (int j=0; j<3; ++j) {
            for (int k=0; k<K_nums; ++k) {
                if (j==0) {
                    theta[0][l][j][k] = gamma1(engine);
                }
                else if (j==1) {
                    theta[0][l][j][k] = gauss(engine);
                }
                else if (j==2) {
                    theta[0][l][j][k] = gamma2(engine);
                }
            }
        }
    }

    // beta_listの定義
    vector<double> beta_vec = make_beta_vec(replica_size);

    // データの定義
    vector<vector<double>> data_vec = generate_data();
    int data_size = 301;
    vector<double> x_data = data_vec[0];
    vector<double> y_data = data_vec[1];

    // mainのループ
    for (int iter=0; iter<iter_nums; ++iter) {
        for (int replica_index=0; replica_index<replica_size; ++replica_index) {
            double beta = beta_vec[replica_index];
            vector<vector<vector<vector<double>>>> next_theta = next_with_metropolis(theta, step_Cd, x_data, y_data, replica_index, beta, K_nums, acceptance_count);

            if (iter >= burn_in) {
                stored_theta[iter - burn_in][replica_index] = next_theta[0][replica_index];
                theta = next_theta;
            }
            else {
                theta = next_theta;
            }
        }
        theta = exchange_replica(theta, x_data, y_data, beta_vec, replica_size, K_nums, exchange_count);
        // プログレス表示
        if (iter % 100 == 0) {
            cout << iter << " end" << endl;
        }
    }
    cout << "### main loop end ###" << endl;

    // 採用率，　交換率を出力
    for (int r=0; r<replica_size; ++r) {
    cout << "交換率" << r << "= " << exchange_count[r]/(iter_nums) << endl;
    }
    cout << "--採用率--" << endl;
    for (int j=0; j<3; ++j) {
        for (int l=0; l<replica_size; ++l) {
            cout << "L"<< l+1 << "_" << j << "=" << (double)(acceptance_count[l][j] / (iter_nums * K_nums)) << endl;
        }
        cout << "\n" << endl;
    }
    // 周辺尤度と自由エネルギーの出力
    double likelihood = culc_z(K_nums, iter_nums, burn_in, replica_size, stored_theta, x_data, y_data, beta_vec);
    cout << "likelihood= " << likelihood << endl;
    cout << "free energy= " << -log(likelihood) << endl;

    //　予測分布を取得
    vector<vector<vector<double>>> q_dist((iter_nums-burn_in), vector<vector<double>>(3, vector<double>(K_nums)));
    
    for (int l=0; l<(iter_nums-burn_in); ++l) {
        for (int p=0; p<3; ++p) {
            for (int q=0; q<K_nums; ++q) {
                q_dist[l][p][q] = stored_theta[l][replica_size-1][p][q];
            }
        }
    }

    // 結果をファイル出力
    string filename_a = "/Users/hayashiyui/lab_tutorial/tutrial4/q_dist/a_dist.txt";
    string filename_mu = "/Users/hayashiyui/lab_tutorial/tutrial4/q_dist/mu_dist.txt";
    string filename_b = "/Users/hayashiyui/lab_tutorial/tutrial4/q_dist/b_dist.txt";
    string file_name;

    for (int j=0; j<3; ++j) {
        if (j == 0) {
        file_name = filename_a;
        }
        else if (j == 1) {
        file_name = filename_mu;
        }
        else if (j == 2) {
        file_name = filename_b;
        }
        ofstream ofs(file_name);

        for (int k=0; k<K_nums; ++k) {
            for (int i=0; i<iter_nums-burn_in; ++i) {
                ofs << to_string(q_dist[i][j][k]) << endl;
            }
        }
    }

    cout << "DONE" << endl;
    return 0;
}
