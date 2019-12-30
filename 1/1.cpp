#include <iostream>
#include <iomanip>
#include <cstdlib>


using namespace std;

const int num_of_lines = 5;
const int num_of_columns = 8;

float find_min_max(float mtr[num_of_lines][num_of_columns]) {
    float mins[num_of_lines];

    #pragma omp parallel for shared(mins, mtr)
    for (int i = 0; i < num_of_lines; ++i)
        mins[i] = mtr[i][0];

    #pragma omp parallel for
    for (int i = 0; i < num_of_lines; ++i) {
        for (int j = 0; j < num_of_columns; ++j) {
        #pragma omp critical
            {	if (mtr[i][j] < mins[i])
                    mins[i] = mtr[i][j];
            }
        }
    }

    float max_of_mins = mins[0];

    #pragma omp parallel for shared(mins, max_of_mins)
    for (int i = 0; i < num_of_lines; ++i) {
        #pragma omp critical
        if (mins[i] > max_of_mins)
            max_of_mins = mins[i];
    }
    return max_of_mins;
}

void print_matrix(float a[num_of_lines][num_of_columns]) {
    for (int i = 0; i < num_of_lines; ++i) {
        for (int j = 0; j < num_of_columns; ++j)
            cout << setw(2) << a[i][j] << " ";
        cout << endl;
    }
}

int main() {
    float matrix[num_of_lines][num_of_columns];
    srand(1);
    for (int i = 0; i < num_of_lines; ++i) {
        for (int j = 0; j < num_of_columns; ++j) {
            matrix[i][j] = rand()%100;
        }
    }

    print_matrix(matrix);
    float min_max = find_min_max(matrix);
    cout << endl << "Min max element is " << min_max << endl;

    return 0;
}

