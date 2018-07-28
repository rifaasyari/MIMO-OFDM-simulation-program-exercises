#define PI 3.1415926535897932
#define ROOT_2 1.414213562
#include <omp.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <string>
#include <vector>
#include <array>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

vector<int> plusOne(vector<int>& digits);
double gaussgen(double mean,double stddev);
void channel_gen(MatrixXcd &channel_H, int Nt, int Nr);
void ML_detection(vector<int>& index, MatrixXcd &channel_H, MatrixXcd &detect, MatrixXcd &detect_y, MatrixXcd &optimal_detection, MatrixXcd &receive_symbol, double *min_distance, MatrixXcd &constellation, int Nt, int Nr);

int main()
{
    int thread_len;
    //thread_id = omp_get_num_procs();
    thread_len = 8;
    int N = 1000,
        Nt = 2,
        Nr = 2,




    srand(time(NULL));   //srand
    //處理器非共用變數
    double min_distance[8];

    array<MatrixXcd, 8> channel_H, tx_symbol, receive_symbol, detect, detect_y, optimal_detection;

    vector<vector<int> > index(8, vector<int>(Nt + 1));

    int thread_i;
    for(unsigned int thread_ii = 0 ; thread_ii < thread_len ; thread_ii++){
        channel_H[thread_ii] = MatrixXd::Zero(Nr,Nt);
        tx_symbol[thread_ii] = MatrixXd::Zero(Nt,1);
        receive_symbol[thread_ii] = MatrixXd::Zero(Nr,1);
        detect[thread_ii] = MatrixXd::Zero(Nt,1);
        detect_y[thread_ii] = MatrixXd::Zero(Nt,1);
        optimal_detection[thread_ii] = MatrixXd::Zero(Nt,1);
        min_distance[thread_ii] = 99999.0;
        for(unsigned int vec_i = 0 ; vec_i < Nt+1 ; vec_i++){
            index[thread_ii][vec_i] = 0;
        }

    }
//----------------start------------------------------------
	//fstream QAM_mimo_2x2;
	//QAM_mimo_2x2.open("QAM_mimo_2x2.txt",ios::out);
	//處理器共用變數
    double snr_db[13] = {0.0},
        snr[13] = {0.0},
        ber[13] = {0.0},
        N0[13] = {0.0};


    MatrixXcd constellation(1,16);
    constellation << 1.0 + 1.0i, 1.0 + 3.0i, 3.0 + 1.0i, 3.0 + 3.0i, -1.0 + 1.0i, -1.0 + 3.0i, -3.0 + 1.0i, -3.0 + 3.0i,
                    -1.0 - 1.0i, -1.0 - 3.0i, -3.0 - 1.0i, -3.0 - 3.0i, 1.0 - 1.0i, 1.0 - 3.0i, 3.0 - 1.0i, 3.0 - 3.0i;


    int len_constellation = 16;
    int K = 4;
    int len_snr = 13;

    int error[len_snr] = {0};

    for(unsigned int i = 0 ; i < len_snr ; i++){
        snr_db[i] = 0.5 + 2.5 * i;
        snr[i] = pow(10.0, snr_db[i] / 10.0);
    }
    double energy = 0.0;
    energy = constellation.squaredNorm();
    double Es = energy / (double)len_constellation;
    double Eb = Es / (double)K;

    clock_t t;
    //#pragma omp parallel
t = clock();



#pragma omp parallel for private(thread_i)
    for(unsigned int i = 0 ; i < len_snr; i++){





        thread_i = omp_get_thread_num();
        //error[i] = 0;

        //#pragma omp parallel for
        for(unsigned int j = 0 ; j < N ; j++){

            index[thread_i][0] = 0;
            min_distance[thread_i] = 9999999;
            N0[i] = Eb * Nr / snr[i];

            for(unsigned int m = 0 ; m < Nt ; m++){
                tx_symbol[thread_i](m, 0) = constellation(0,rand()%16);
            }

            channel_gen(channel_H[thread_i], Nt, Nr);

            receive_symbol[thread_i] = channel_H[thread_i] * tx_symbol[thread_i];
            for(unsigned int r_i = 0 ; r_i < Nr ; r_i++){
                receive_symbol[thread_i].real()(r_i,0) += gaussgen(0, sqrt(N0[i]/2));
                receive_symbol[thread_i].imag()(r_i,0) += gaussgen(0, sqrt(N0[i]/2));
            }


            ML_detection(index[thread_i], channel_H[thread_i], detect[thread_i], detect_y[thread_i], optimal_detection[thread_i], receive_symbol[thread_i], &min_distance[thread_i], constellation, Nt, Nr);


            for(unsigned int err_i = 0 ; err_i < Nt ; err_i++){
                //int test_real = fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).real()(err_i, 0));
                //int test_imag = fabs((optimal_detection[thread_i] - tx_symbol[thread_i]).imag()(err_i, 0));

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



}
//snr forloop end
//parallel end

for(unsigned int i = 0; i < 13 ; i++){
        cout << " snr_db: "<< snr_db[i] << " ber =  " << ber[i]  << endl << endl;
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




vector<int> plusOne(vector<int>& digits) {
    if (digits.empty()) return digits;
    int carry = 1, n = digits.size();

    for (unsigned int i = n - 1; i >= 0; --i) {
        if (carry == 0) return digits;
        int sum = digits[i] + carry;
        digits[i] = sum % 16;
        carry = sum / 16;
    }
    if (carry == 1) digits.insert(digits.begin(), 1);

    return digits;
}



