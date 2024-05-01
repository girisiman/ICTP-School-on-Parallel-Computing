#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_N 1000  // Maximum dimension of the matrix
#define NUM_RUNS 5   // Number of runs for the experiment

double getTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int main() {
    int N = 100;  // Matrix dimension (adjust as needed)
    double execution_times[NUM_RUNS];

    // Perform the experiment multiple times
    for (int run_idx = 0; run_idx < NUM_RUNS; run_idx++) {
        // Initialize and generate a random matrix
        float A[MAX_N][MAX_N + 1];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= N; j++) {
                A[i][j] = (float)(rand()) / RAND_MAX * 10.0;
            }
        }

        // Measure execution time using OpenMP
        double startTime = getTimeInSeconds();

        // Perform Gaussian elimination with OpenMP parallelization
        #pragma omp parallel
        {
            double thread_start = getTimeInSeconds();
            #pragma omp for collapse(2)
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    if (i > j) {
                        float c = A[i][j] / A[j][j];
                        for (int k = 0; k <= N; k++) {
                            A[i][k] -= c * A[j][k];
                        }
                    }
                }
            }
            double thread_end = getTimeInSeconds();
            int tid = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            printf("Thread %d of %d: Execution Time = %.6f seconds\n", tid, num_threads, thread_end - thread_start);
        }

        // Back substitution (not parallelized for simplicity)
        for (int i = N - 1; i >= 0; i--) {
            float sum = 0.0;
            for (int j = i + 1; j < N; j++) {
                sum += A[i][j] * A[j][N];
            }
            A[i][N] = (A[i][N] - sum) / A[i][i];
        }

        double endTime = getTimeInSeconds();
        double elapsedTime = endTime - startTime;

        // Store the overall execution time
        execution_times[run_idx] = elapsedTime;

        // Print the execution time for this run
        printf("Run %d: Overall Execution Time = %.6f seconds\n", run_idx + 1, elapsedTime);
    }

    // Save execution times to a text file
    FILE *fp = fopen("execution_times.txt", "w");
    if (fp == NULL) {
        printf("Error: Unable to open file for writing.\n");
        return 1;
    }

    // Write execution times to the file
    for (int i = 0; i < NUM_RUNS; i++) {
        fprintf(fp, "%.6f\n", execution_times[i]);
    }

    // Close the file
    fclose(fp);

    return 0;
}
