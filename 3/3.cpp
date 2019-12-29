#include <iostream>
#include <omp.h>
#include <ctime>
#include <iomanip>
#include <cstdlib>


using namespace std;

const int N = 8;
const int num_of_threads = 8;

void rand_init(float** Matr, float* vect, float* result_vect) {
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Matr[i][j] = rand() % 10;
        }
        vect[i] = rand() % 10;
        result_vect[i] = 0;
    }
}

void print_matrix(float** Matr, int dim1, int dim2) {
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j)
            cout << setw(1) << Matr[i][j] << " ";
        cout << endl;
    }
}

void print_vector(float* vec, int dim) {
    for (int i = 0; i < dim; ++i) {
        cout << setw(2) << vec[i] << " ";
    }
    cout << endl;
}

double serial_calculate(float** Matr, float* vect, float* result_vect) {
    int i, j;
    double time = omp_get_wtime();
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j)
            result_vect[i] += Matr[i][j] * vect[j];
    }
    return omp_get_wtime() - time;
}

// Data separation by rows
double parallel_raw_calculate(float** Matr, float* vect, float* result_vect) {
    int i, j;
    float sum = 0;
    omp_set_num_threads(num_of_threads);
    double time = omp_get_wtime();
    for (i = 0; i < N; ++i) {
        sum = 0;
    #pragma omp paralell for redunction(+:sum)
        for (j = 0; j < N; ++j)
            sum += Matr[i][j] * vect[j];
        result_vect[i] = sum;
    }
    return omp_get_wtime() - time;
}

double parallel_column_calculate(float** Matr, float* vect, float* result_vect) {
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

double parallel_block_calculation(float** Matr, float* vect, float* result_vect) {
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


int main() {
    auto** Matrix = new float*[N];
    for(int i = 0 ; i < N ; ++i)
        Matrix[i] = new float[N];
    auto* vect = new float[N];
    auto* serial_result = new float[N];
    auto* row_result = new float[N];
    auto* column_result = new float[N];
    auto* block_result = new float[N];

    rand_init(Matrix, vect, serial_result);
    double serial_time = serial_calculate(Matrix, vect, serial_result);
    double parallel_raw = parallel_raw_calculate(Matrix, vect, row_result);
    double parallel_column = parallel_column_calculate(Matrix, vect, column_result);
    double parallel_block = parallel_block_calculation(Matrix, vect, block_result);

    cout  << "Matrix:" << endl;
    print_matrix(Matrix, N, N);
    cout << "Vector:" << endl;
    print_vector(vect, N);
    cout << "Serial result: (time: " << serial_time << ")" << endl;
    print_vector(serial_result, N);
    cout << "Raw result: (time: " << parallel_raw << ")" << endl;
    print_vector(row_result, N);
    cout << "Colomn result: (time: " << parallel_column << ")" << endl;
    print_vector(column_result, N);
    cout << "Block result: (time: " << parallel_block << ")" << endl;
    print_vector(block_result, N);
}
