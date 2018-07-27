#define PI 3.1415926535897932
#define ROOT_2 1.414213562


#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <string>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

double gaussgen(double mean,double stddev);
void channel_gen(MatrixXcd &channel_H, int Nt, int Nr);
void ML_detection(MatrixXcd &channel_H, MatrixXcd &detect, MatrixXcd &optimal_detection, MatrixXcd &receive_symbol, int current, double *min_distance, MatrixXcd &constellation, int Nt, int Nr);

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
    int N = 10000,
        Nt = 2,
        Nr = 2,
        error = 0;

    MatrixXcd channel_H,
        tx_symbol,
        receive_symbol,
        detect,
        optimal_detection;

    channel_H = MatrixXd::Zero(Nr,Nt);
    tx_symbol = MatrixXd::Zero(Nt,1);
    receive_symbol = MatrixXd::Zero(Nr,1);
    detect = MatrixXd::Zero(Nt,1);
    optimal_detection = MatrixXd::Zero(Nt,1);

    MatrixXcd constellation(1,16);
    constellation << 1.0 + 1.0i, 1.0 + 3.0i, 3.0 + 1.0i, 3.0 + 3.0i, -1.0 + 1.0i, -1.0 + 3.0i, -3.0 + 1.0i, -3.0 + 3.0i,
                    -1.0 - 1.0i, -1.0 - 3.0i, -3.0 - 1.0i, -3.0 - 3.0i, 1.0 - 1.0i, 1.0 - 3.0i, 3.0 - 1.0i, 3.0 - 3.0i;
    int len_constellation = 16;
    int K = 4;
    int len_snr = 13;


    for(int i = 0 ; i < len_snr ; i++){
        snr_db[i] = 0.5 + 2.5 * (i+1);
        snr[i] = pow(10.0, snr_db[i] / 10.0);

    }
    double energy = 0.0;
    energy = constellation.squaredNorm();
    double Es = (double)energy / len_snr;
    double Eb = (double)Es / K;





    clock_t t;
    for(int i = 0 ; i < len_snr; i++){
        error = 0;
        t = clock();
        N0 = Eb * Nr / snr[i];
        for(int j = 0 ; j < N ; j++){

                for(int m = 0 ; m < Nt ; m++){
                    tx_symbol(m, 0) = constellation(0,rand()%16);
                }

                //cout << tx_symbol << endl << endl;

                channel_gen(channel_H, Nt, Nr);

                receive_symbol = channel_H * tx_symbol;
                for(int r_i = 0 ; r_i < Nr ; r_i++){
                    receive_symbol.real()(r_i,0) += gaussgen(0, sqrt(N0)/sqrt(2));
                    receive_symbol.imag()(r_i,0) += gaussgen(0, sqrt(N0)/sqrt(2));
                }
                //cout << receive_symbol << endl << endl;
                min_distance = 99999;
                ML_detection(channel_H, detect, optimal_detection, receive_symbol, 0, &min_distance, constellation, Nt, Nr);
                //cout << optimal_detection << endl << endl;
                //cout << detect << endl << endl;

                for(int err_i = 0 ; err_i < Nt ; err_i++){
                    int test_real = fabs((optimal_detection - tx_symbol).real()(err_i, 0));
                    int test_imag = fabs((optimal_detection - tx_symbol).imag()(err_i, 0));

                    if(test_real == 2 || test_real == 6){
                        error += 1;
                    }
                    if(test_real == 4){
                        error += 2;
                    }
                    if(test_imag == 2 || test_imag == 6){
                        error += 1;
                    }
                    if(test_imag == 4){
                        error += 2;
                    }
                    //cout << test_real  << endl << endl;
                }




        }
        ber[i] = (double)error / (K * Nt * N );
        cout << ber[i]  << endl << endl;
        t = clock() - t;
        cout << "time elapsed " << (double) t/(double)CLOCKS_PER_SEC << "sec " << endl;


    }



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




void ML_detection(MatrixXcd &channel_H, MatrixXcd &detect, MatrixXcd &optimal_detection, MatrixXcd &receive_symbol, int current, double *min_distance, MatrixXcd &constellation, int Nt, int Nr){

    if(current == Nt){
        MatrixXcd detect_y = MatrixXd::Zero(Nr,1);
        detect_y = channel_H * detect;
        double dis = 0.0;
        dis = (receive_symbol - detect_y).squaredNorm();

        if(dis < *min_distance){
            *min_distance = dis;
            optimal_detection = detect;

        }


    }
    else{

        for(int i = 0 ; i < 16; i++){
            detect(current, 0) = constellation(0, i);
            ML_detection(channel_H, detect, optimal_detection, receive_symbol, current+1, min_distance, constellation, Nt, Nr);
        }

    }


}


void channel_gen(MatrixXcd &channel_H, int Nt, int Nr){
    for(int h_i = 0 ; h_i < Nr ; h_i++){
        for(int h_j = 0 ; h_j < Nt ; h_j++){
            channel_H.real()(h_i, h_j) = gaussgen(0, 1/sqrt(2));
            channel_H.imag()(h_i, h_j) = gaussgen(0, 1/sqrt(2));
        }
    }
}
