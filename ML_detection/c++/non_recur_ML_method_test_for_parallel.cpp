#define PI 3.1415926535897932
#define ROOT_2 1.414213562
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

vector<int> plusOne(vector<int>& digits);
double gaussgen(double mean,double stddev);
void channel_gen(MatrixXcd &channel_H, int Nt, int Nr);
void ML_detection(vector<int>& index, MatrixXcd &channel_H, MatrixXcd &detect, MatrixXcd &detect_y, MatrixXcd &optimal_detection, MatrixXcd &receive_symbol, double *min_distance, MatrixXcd &constellation, int Nt, int Nr);

int main()
{
    srand(time(NULL));   //srand

//處理器共用變數
    const int thread_len = omp_get_num_procs(),
        snr_len = 16,
        N = 1e6,
        Nt = 1,
        Nr = 1;
    MatrixXcd constellation(1,16);
    constellation << 1.0 + 1.0i, 1.0 + 3.0i, 3.0 + 1.0i, 3.0 + 3.0i, -1.0 + 1.0i, -1.0 + 3.0i, -3.0 + 1.0i, -3.0 + 3.0i,
                    -1.0 - 1.0i, -1.0 - 3.0i, -3.0 - 1.0i, -3.0 - 3.0i, 1.0 - 1.0i, 1.0 - 3.0i, 3.0 - 1.0i, 3.0 - 3.0i;
    const int len_constellation = 16;
    const int K = log2(len_constellation);
    double energy = constellation.squaredNorm();
    double Es = energy / (double)len_constellation;
    double Eb = Es / (double)K;

//處理器非共用參數，這裡創了數量為核心數的倍數空間，代替了平行執行完後空間釋放問題
    int thread_i;
    vector<double> min_distance(thread_len, 99999.0);
    vector<MatrixXcd> channel_H(thread_len), tx_symbol(thread_len), receive_symbol(thread_len), detect(thread_len), detect_y(thread_len), optimal_detection(thread_len);
    vector<vector<int> > index(thread_len, vector<int>(Nt + 1, 0));
    //matrix的初始化
    for(unsigned int thread_ii = 0 ; thread_ii < thread_len ; thread_ii++){
        channel_H[thread_ii] = MatrixXd::Zero(Nr,Nt);
        tx_symbol[thread_ii] = MatrixXd::Zero(Nt,1);
        receive_symbol[thread_ii] = MatrixXd::Zero(Nr,1);
        detect[thread_ii] = MatrixXd::Zero(Nt,1);
        detect_y[thread_ii] = MatrixXd::Zero(Nt,1);
        optimal_detection[thread_ii] = MatrixXd::Zero(Nt,1);
    }
    double snr_db[snr_len] = {0.0},
        snr[snr_len] = {0.0},
        ber[snr_len] = {0.0},
        N0[snr_len] = {0.0};
    int error[snr_len] = {0};
    for(unsigned int i = 0 ; i < snr_len ; i++){
        snr_db[i] =  1.8 * i ;
        snr[i] = pow(10.0, snr_db[i] / 10.0);
        //這裡乘上Nr為normalization，代表每個接收天線能量相同，有Nr個接收天線
        N0[i] = Eb * Nr / snr[i];
    }


//----------------start------------------------------------
//程式forloop時間計算開始
    clock_t t;
    t = clock();

	fstream mimo_16QAM_2x2;
	mimo_16QAM_2x2.open("mimo_16QAM_1x1.txt",ios::out);

	//平行處理snr資料
    #pragma omp parallel for private(thread_i)
    for(unsigned int i = 0 ; i < snr_len; i++){

        //獲取處理編號
        thread_i = omp_get_thread_num();

        for(unsigned int j = 0 ; j < N ; j++){

            //index的0位置為存放ML是否搜尋完畢，0為未搜完，1為已搜完
            //其他位置存放天線的星座點位置編號
            index[thread_i][0] = 0;
            min_distance[thread_i] = 9999999.0;

            //建立要傳送的隨機symbol
            for(unsigned int m = 0 ; m < Nt ; m++){
                tx_symbol[thread_i](m, 0) = constellation(0,rand()%16);
            }
            //建立通道矩陣
            channel_gen(channel_H[thread_i], Nt, Nr);

            //建立接收symbol
            receive_symbol[thread_i] = channel_H[thread_i] * tx_symbol[thread_i];
            for(unsigned int r_i = 0 ; r_i < Nr ; r_i++){
                receive_symbol[thread_i].real()(r_i,0) += gaussgen(0, sqrt(N0[i]/2));
                receive_symbol[thread_i].imag()(r_i,0) += gaussgen(0, sqrt(N0[i]/2));
            }

            //偵測symbol
            ML_detection(index[thread_i], channel_H[thread_i], detect[thread_i], detect_y[thread_i], optimal_detection[thread_i], receive_symbol[thread_i], &min_distance[thread_i], constellation, Nt, Nr);

            //偵測symbol和傳送symbol的錯誤數量計算，gray code
            for(unsigned int err_i = 0 ; err_i < Nt ; err_i++){

                if(fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).real()(err_i, 0)) == 2 || fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).real()(err_i, 0)) == 6){
                    error[i] += 1;
                }
                else if(fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).real()(err_i, 0)) == 4){
                    error[i] += 2;
                }
                if(fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).imag()(err_i, 0)) == 2 || fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).imag()(err_i, 0)) == 6){
                    error[i] += 1;
                }
                else if(fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).imag()(err_i, 0)) == 4){
                    error[i] += 2;
                }
            }
        }
        ber[i] = (double)error[i] / (double)(K * Nt * N );

        //printf("loop %d, thread %d\n", i, thread_i);

    }
    t = clock() - t;
    //snr forloop end
    //parallel process end

    mimo_16QAM_2x2 << "snr_db" << "           " << "ber" << endl;
    for(unsigned int i = 0; i < snr_len ; i++){

        mimo_16QAM_2x2 << setw(6) << fixed << setprecision(1) <<  snr_db[i] << "    "  << setprecision(8) << ber[i] << endl;
        //cout << " snr_db: "<< snr_db[i] << " ber =  " << ber[i]  << endl;

    }

    double total_sec = (double)t / CLOCKS_PER_SEC;
    int t_sec = (int)total_sec % 60;
    total_sec /= 60;
    int t_min = (int)total_sec % 60;
    total_sec /= 60;
    int t_hour = (int)total_sec % 24;
    total_sec /= 24;
    int t_day = (int)total_sec % 24;

    mimo_16QAM_2x2 << endl;
    mimo_16QAM_2x2 << "time elapsed " << t_day << "day " << t_hour << "hour " << t_min << "min " << t_sec << "sec " << endl;
    //cout << "time elapsed " << t_day << "day " << t_hour << "hour " << t_min << "min " << t_sec << "sec " << endl;

    mimo_16QAM_2x2.close();
    return 0;
}


double gaussgen(double mean, double stddev)
{//Box muller method
    static double n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached)
    {
        double x, y, r;
        do
        {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        }
        while (r == 0.0 || r > 1.0);
        {
            double d = sqrt(-2.0*log(r)/r);
            double n1 = x*d;
            n2 = y*d;
            double result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    }
    else
    {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}


void ML_detection(vector<int>& index, MatrixXcd &channel_H, MatrixXcd &detect, MatrixXcd &detect_y, MatrixXcd &optimal_detection, MatrixXcd &receive_symbol, double *min_distance, MatrixXcd &constellation, int Nt, int Nr){

    while(index[0] != 1){

        for(unsigned int j = 1 ; j < Nt+1 ; j++){

            detect(j-1, 0) = constellation(0, index[j]);
        }
        detect_y = channel_H * detect;
        double dis = 0.0;
        dis = (receive_symbol - detect_y).squaredNorm();

        if(dis < *min_distance){
            *min_distance = dis;
            optimal_detection = detect;
        }
        plusOne(index);
    }
}

void channel_gen(MatrixXcd &channel_H, int Nt, int Nr){
    for(unsigned int h_i = 0 ; h_i < Nr ; h_i++){
        for(unsigned int h_j = 0 ; h_j < Nt ; h_j++){
            channel_H.real()(h_i, h_j) = gaussgen(0, 1/ROOT_2);
            channel_H.imag()(h_i, h_j) = gaussgen(0, 1/ROOT_2);
        }
    }
}

vector<int> plusOne(vector<int>& digits){
    if(digits.empty()) return digits;
    int carry = 1, n = digits.size();

    for(int i = n - 1; i >= 0; --i) {
        if (carry == 0) return digits;
        int sum = digits[i] + carry;
        digits[i] = sum % 16;
        carry = sum / 16;
    }
    if(carry == 1) digits.insert(digits.begin(), 1);

    return digits;
}
