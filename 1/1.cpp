#include <iostream>
#include <omp.h>
#include <ctime>
#include <iomanip>
#include <cstdlib>


using namespace std;

const unsigned int DIM1 = 3;  // rows number
const unsigned int DIM2 = 5;  //columns number

float min_max_element(float mtr[DIM1][DIM2])
{
	float mins[DIM1];

#pragma omp parallel for shared(mins, mtr)
	for (int i = 0; i < DIM1; i++)
		mins[i] = mtr[i][0];

#pragma omp parallel for
	for (int i = 0; i < DIM1; i++) {
		for (int j = 0; j < DIM2; j++) {
#pragma omp critical
			{	if (mtr[i][j] < mins[i])
					mins[i] = mtr[i][j];
				int myID = omp_get_thread_num();
				int threads = omp_get_num_threads();
				std::cout << "Num of thread in inner loop is " << myID << " from " << threads << endl;
			}
		}
		int myID = omp_get_thread_num();
		int threads = omp_get_num_threads();
		std::cout << "Num of thread in outer loop is " << myID << " from " << threads << endl;

	}

	float max_of_mins = mins[0];

#pragma omp parallel for shared(mins, max_of_mins) 
	for (int i = 0; i < DIM1; i++) 
#pragma omp critical
		{
		if (mins[i] > max_of_mins) {
			max_of_mins = mins[i];
		}
	}	

	return max_of_mins;
}

void print_matrix(float a[DIM1][DIM2]) {

	for (int r = 0; r < DIM1; ++r) {
		for (int c = 0; c < DIM2; ++c)
			cout << setw(5) << a[r][c] << " ";
		cout << endl;
	}
}

int main() {
	float matrix[DIM1][DIM2];

	std::srand(unsigned(std::time(0)));
	//srand(1);
	for (int i = 0; i < DIM1; i++) {
		for (int j = 0; j < DIM2; j++) {
			matrix[i][j] = rand()%100;
		}
	}

	print_matrix(matrix);
	float min_max = min_max_element(matrix);
	cout << endl << "Min_max_element is " << min_max << endl;
	
	return 0;
}

