#define PI 3.1415926535897932
#define ROOT_2 1.414213562
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
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
//----------------start------------------------------------
	//fstream QAM_mimo_2x2;
	//QAM_mimo_2x2.open("QAM_mimo_2x2.txt",ios::out);
    double snr_db[13] = {0.0},
        snr[13] = {0.0},
        ber[13] = {0.0},
        min_distance = 99999.0,
        N0 = 0.0;

    int N = 1000,
        Nt = 2,
        Nr = 2,
        error = 0;

    MatrixXcd channel_H = MatrixXd::Zero(Nr,Nt);
    MatrixXcd tx_symbol = MatrixXd::Zero(Nt,1);
    MatrixXcd receive_symbol = MatrixXd::Zero(Nr,1);
    MatrixXcd detect = MatrixXd::Zero(Nt,1);
    MatrixXcd detect_y = MatrixXd::Zero(Nt,1);
    MatrixXcd optimal_detection = MatrixXd::Zero(Nt,1);
    vector<int> index(Nt+1,0);
    MatrixXcd constellation(1,16);
    constellation << 1.0 + 1.0i, 1.0 + 3.0i, 3.0 + 1.0i, 3.0 + 3.0i, -1.0 + 1.0i, -1.0 + 3.0i, -3.0 + 1.0i, -3.0 + 3.0i,
                    -1.0 - 1.0i, -1.0 - 3.0i, -3.0 - 1.0i, -3.0 - 3.0i, 1.0 - 1.0i, 1.0 - 3.0i, 3.0 - 1.0i, 3.0 - 3.0i;
    int len_constellation = 16;
    int K = 4;
    int len_snr = 13;
    vector<double> max_min_distance(len_snr, 0);

    for(int i = 0 ; i < len_snr ; i++){
        snr_db[i] = 0.5 + 2.5 * i;
        snr[i] = pow(10.0, snr_db[i] / 10.0);
    }
    double energy = 0.0;
    energy = constellation.squaredNorm();
    double Es = energy / (double)len_constellation;
    double Eb = Es / (double)K;

    clock_t t;
    t = clock();
    //#pragma omp parallel
    for(int i = 0 ; i < len_snr; i++){
        error = 0;

        //#pragma omp parallel for
        for(int j = 0 ; j < N ; j++){

            index[0] = 0;
            min_distance = 9999999;
            N0 = Eb / snr[i];

            for(int m = 0 ; m < Nt ; m++){
                tx_symbol(m, 0) = constellation(0,rand()%16);
            }

            channel_gen(channel_H, Nt, Nr);

            receive_symbol = channel_H * tx_symbol;
            for(int r_i = 0 ; r_i < Nr ; r_i++){
                receive_symbol.real()(r_i,0) += gaussgen(0, sqrt(N0/2));
                receive_symbol.imag()(r_i,0) += gaussgen(0, sqrt(N0/2));
            }


            ML_detection(index, channel_H, detect, detect_y, optimal_detection, receive_symbol, &min_distance, constellation, Nt, Nr);


            for(int err_i = 0 ; err_i < Nt ; err_i++){
                int test_real = fabs((optimal_detection - tx_symbol).real()(err_i, 0));
                int test_imag = fabs((optimal_detection - tx_symbol).imag()(err_i, 0));

                if(test_real == 2 || test_real == 6){
                    error += 1;
                }
                else if(test_real == 4){
                    error += 2;
                }
                if(test_imag == 2 || test_imag == 6){
                    error += 1;
                }
                else if(test_imag == 4){
                    error += 2;
                }
            }
        }

        ber[i] = (double)error / (double)(K * Nt * N );


    }

        for(int i = 0; i < 13 ; i++){
        cout << " snr: i, ber =  " << ber[i]  << endl << endl;
    }
    t = clock() - t;
        cout << "time elapsed " << (double) t/(double)CLOCKS_PER_SEC << "sec " << endl;
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

        for(int j = 1 ; j < Nt+1 ; j++){

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
    for(int h_i = 0 ; h_i < Nr ; h_i++){
        for(int h_j = 0 ; h_j < Nt ; h_j++){
            channel_H.real()(h_i, h_j) = gaussgen(0, 1/ROOT_2);
            channel_H.imag()(h_i, h_j) = gaussgen(0, 1/ROOT_2);
        }
    }
}




vector<int> plusOne(vector<int>& digits) {
    if (digits.empty()) return digits;
    int carry = 1, n = digits.size();

    for (int i = n - 1; i >= 0; --i) {
        if (carry == 0) return digits;
        int sum = digits[i] + carry;
        digits[i] = sum % 16;
        carry = sum / 16;
    }
    if (carry == 1) digits.insert(digits.begin(), 1);

    return digits;
}



