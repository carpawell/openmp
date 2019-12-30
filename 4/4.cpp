#include <iostream>
#include <omp.h>
#include <ctime>
#include <iomanip>
#include <cstdlib>
#include <cmath>


using namespace std;

const int N = 1000;
const int num_of_threads = 4;

float** matrix_rand_init(int random) {
    auto ** new_matrix = new float*[N];
    for (int i = 0; i < N; ++i) {
        new_matrix[i] = new float[N];
    }
    srand(random);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            new_matrix[i][j] = rand() % 10;
        }
    }
    return new_matrix;
}

float** matrix_zero_init() {
    auto ** new_matrix = new float*[N];
    for (int i = 0; i < N; ++i) {
        new_matrix[i] = new float[N];
    }
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            new_matrix[i][j] = 0;
        }
    }
    return new_matrix;
}

void print_matrix(float** Matr) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            cout << setw(1) << Matr[i][j] << " ";
        cout << endl;
    }
}

double serial_calculate(float** A, float** B, float** result) {
    int i, j, k;
    double time = omp_get_wtime();
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            result[i][j] = 0;
            for (k = 0; k < N; ++k) {
                result[i][j] = result[i][j] + A[i][k] * B[k][j];
            }
        }
    }
    return omp_get_wtime() - time;
}

double parallel_raw_calculate(float** A, float** B, float** result) {
    int i, j, k;
    float sum = 0;
    omp_set_num_threads(num_of_threads);
    double time = omp_get_wtime();
    #pragma omp paralell for private (j, k)
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return omp_get_wtime() - time;
}

double parallel_tape_sharing_calculate(float** A, float** B, float** result) {
    int i, j, k;
    omp_set_nested(true);
    omp_set_num_threads(num_of_threads);
    double time = omp_get_wtime();
    #pragma omp parallel for private (j, k)
    for (i = 0; i < N; ++i) {
    #pragma omp parallel for private (k)
        for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return omp_get_wtime() - time;
}

double parallel_block_calculate(float** A, float** B, float** result) {
    int grid_size = int(sqrt((double)num_of_threads));
    int block_size = N / grid_size;
    omp_set_num_threads(num_of_threads);
    double time = omp_get_wtime();
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int row = int(thread_id / grid_size);
        int col = thread_id % grid_size;
        for (int iter = 0; iter < grid_size; ++iter) {
            for (int i = row * block_size; i < (row + 1) * block_size; ++i) {
                for (int j = col * block_size; j < (col + 1) * block_size; ++j) {
                    for (int k = iter * block_size; k < (iter + 1) * block_size; ++k) {
                        result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
    }
    return omp_get_wtime() - time;
}


int main() {
    float** A = matrix_rand_init(1);
    cout << "Matrix A:" << endl;
//    print_matrix(A);
    float** B = matrix_rand_init(2);
    cout << "Matrix B:" << endl;
//    print_matrix(B);
    float** serial_C = matrix_zero_init();
    float** parallel_raw_C = matrix_zero_init();
    float** parallel_tape_C = matrix_zero_init();
    float** parallel_block_C = matrix_zero_init();

    double serial_time = serial_calculate(A, B, serial_C);
    double parallel_raw_time = parallel_raw_calculate(A, B, parallel_raw_C);
    double parallel_tape_time = parallel_tape_sharing_calculate(A, B, parallel_tape_C);
    double parallel_block_time = parallel_block_calculate(A, B, parallel_block_C);

//    print_matrix(serial_C);
//    print_matrix(parallel_raw_C);
//    print_matrix(parallel_tape_C);
//    print_matrix(parallel_block_C);
    cout << "Serial results: " << serial_time << endl;
    cout << "Parallel raw results: " << parallel_raw_time << endl;
    cout << "Parallel tape results: " << parallel_tape_time << endl;
    cout << "Parallel block results: " << parallel_block_time << endl;
}
