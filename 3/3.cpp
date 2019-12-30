#include <iostream>
#include <omp.h>
#include <ctime>
#include <iomanip>
#include <cstdlib>


using namespace std;

const int N = 8;
int num_of_threads = 8;


void print_matrix(float** Matrix) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            cout << setw(1) << Matrix[i][j] << " ";
        cout << endl;
    }
}

void print_vector(float* vector) {
    for (int i = 0; i < N; ++i) {
        cout << setw(2) << vector[i] << " ";
    }
    cout << endl;
}

double serial_calculate(float** Matr, const float* vect, float* result_vect) {
    int i, j;
    double time = omp_get_wtime();
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j)
            result_vect[i] += Matr[i][j] * vect[j];
    }
    return omp_get_wtime() - time;
}

double parallel_raw_calculate(float** Matr, const float* vect, float* result_vect) {
    int i, j;
    float sum = 0;
    omp_set_num_threads(num_of_threads);
    double time = omp_get_wtime();
    for (i = 0; i < N; ++i) {
        sum = 0;
    #pragma omp paralell for redunction(+ : sum)
        for (j = 0; j < N; ++j)
            sum += Matr[i][j] * vect[j];
        result_vect[i] = sum;
    }
    return omp_get_wtime() - time;
}

double parallel_column_calculate(float** Matr, const float* vect, float* result_vect) {
    omp_set_num_threads(num_of_threads);
    auto** partial_sums = new float*[N];
    for(int i = 0 ; i < N ; ++i)
        partial_sums[i] = new float[N];
    double time = omp_get_wtime();
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int block_size = N / num_of_threads;
        float IterResult;
        for (int i = 0; i < N; ++i) {
            IterResult = 0;
            for (int j = 0; j < block_size; ++j)
                IterResult += Matr[i][j +
                                      thread_id*block_size] *
                              vect[j + thread_id*block_size];
            partial_sums[thread_id][i] = IterResult;
        }
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < num_of_threads; ++j)
            result_vect[i] += partial_sums[j][i];
    return omp_get_wtime() - time;
}

double parallel_block_calculation(float** Matr, const float* vect, float* result_vect) {
    omp_set_num_threads(num_of_threads);
    omp_set_nested(true);
    double time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float thread_sum = 0;
    #pragma omp parallel for reduction(+:thread_sum)
        for (int j = 0; j < N; ++j)
            thread_sum += Matr[i][j] * vect[j];
        result_vect[i] = thread_sum;
    }
    return omp_get_wtime() - time;
}

void init_with_random_values(float** Matr, float* vect) {
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Matr[i][j] = rand() % 10;
        }
        vect[i] = rand() % 10;
    }
}


int main() {
    auto** A = new float*[N];
    for(int i = 0 ; i < N ; ++i)
        A[i] = new float[N];
    auto* b = new float[N];
    auto* serial_result = new float[N];
    auto* row_result = new float[N];
    auto* column_result = new float[N];
    auto* block_result = new float[N];

    init_with_random_values(A, b);

    for (int i = 2; i <= 8; ++i) {
        num_of_threads = i;
        double serial_time = serial_calculate(A, b, serial_result);
        double parallel_raw = parallel_raw_calculate(A, b, row_result);
        double parallel_column = parallel_column_calculate(A, b, column_result);
        double parallel_block = parallel_block_calculation(A, b, block_result);
        cout << "Results for " << i << " threads:" << endl;
//        cout  << "A:" << endl;
//        print_matrix(A);
//        cout << "Vector:" << endl;
//        print_vector(b);
        cout << "Serial result: " << serial_time << ")" << endl;
//        print_vector(serial_result);
        cout << "Raw result: " << parallel_raw << ")" << endl;
//        print_vector(row_result);
        cout << "Colomn result: " << parallel_column << ")" << endl;
//        print_vector(column_result);
        cout << "Block result: " << parallel_block << ")" << endl;
//        print_vector(block_result);
        cout << "===================================" << endl;
    }
}
