#include <iostream>
#include <iomanip>
#include <cstdlib>


using namespace std;

const int N = 3;

void print_matrix(double a[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            cout << setw(2) << a[i][j] << " ";
        cout << endl;
    }
}

int main() {
    double a[N][N];
    double b[N][N];
    double c[N][N];

    srand(1);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 10;
            b[i][j] = rand() % 10;
            c[i][j] = 0;
        }

    #pragma omp parallel for shared(a, b)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    cout << "Matrix A:" << endl;
    print_matrix(a);
    cout << endl << "Matrix B:" << endl;
    print_matrix(b);
    cout << endl << "Matrix C = A * B:" << endl;
    print_matrix(c);
    return 0;
}
