#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>  // Include OpenMP header

#define MAX_N 1000  // Maximum dimension of the matrix
#define NUM_SIZES 5  // Number of different matrix sizes to test
#define NUM_RUNS 5   // Number of runs for each matrix size

double getTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int main() {
    int N_values[NUM_SIZES] = {100, 200, 300, 400, 500};  // Matrix sizes to test
    double execution_times[NUM_SIZES][NUM_RUNS];
    float A[MAX_N][MAX_N + 1];
    float x[MAX_N];
    float c, sum;

    srand(time(NULL));

    // Perform experiments for each matrix size
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int N = N_values[size_idx];

        // Run multiple times for each matrix size
        for (int run_idx = 0; run_idx < NUM_RUNS; run_idx++) {
            // Generate a random augmented matrix
            #pragma omp parallel for shared(A) private(c)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j <= N; j++) {
                    A[i][j] = (float)(rand()) / RAND_MAX * 10.0;
                }
            }

            // Measure execution time
            double startTime = getTimeInSeconds();

            // Perform Gaussian elimination with OpenMP parallelization
            for (int j = 0; j < N; j++) {
                #pragma omp parallel for shared(A, j) private(c, sum)
                for (int i = j + 1; i < N; i++) {
                    c = A[i][j] / A[j][j];
                    for (int k = j; k <= N; k++) {
                        A[i][k] -= c * A[j][k];
                    }
                }
            }

            // Back substitution to find the solution
            for (int i = N - 1; i >= 0; i--) {
                sum = 0;
                for (int j = i + 1; j < N; j++) {
                    sum += A[i][j] * x[j];
                }
                x[i] = (A[i][N] - sum) / A[i][i];
            }

            double endTime = getTimeInSeconds();
            double elapsedTime = endTime - startTime;

            // Store the execution time
            execution_times[size_idx][run_idx] = elapsedTime;
        }
    }

    // Write results to a text file
    FILE *fp = fopen("execution_times.txt", "w");
    if (fp == NULL) {
        printf("Error: Unable to open file.\n");
        return 1;
    }

    // Write matrix sizes and corresponding average execution times
    for (int size_idx = 0; size_idx < NUM_SIZES; size_idx++) {
        int N = N_values[size_idx];
        double avg_time = 0.0;

        // Calculate average time for each matrix size
        for (int run_idx = 0; run_idx < NUM_RUNS; run_idx++) {
            avg_time += execution_times[size_idx][run_idx];
        }
        avg_time /= NUM_RUNS;

        // Write to file
        fprintf(fp, "%d %.6f\n", N, avg_time);
    }

    // Close file
    fclose(fp);

    return 0;
}
