#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAX_N 1000  // Maximum dimension of the matrix
#define NUM_RUNS 10  // Number of runs for each matrix size

double getTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

int main() {
    int N = 1000;  // Fixed matrix size for the experiment
    double execution_times[NUM_RUNS];
    float A[MAX_N][MAX_N + 1];
    float x[MAX_N];
    float c, sum;

    srand(time(NULL));

    // Perform the experiment multiple times
    for (int run_idx = 0; run_idx < NUM_RUNS; run_idx++) {
        // Generate a random augmented matrix
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= N; j++) {
                A[i][j] = (float)(rand()) / RAND_MAX * 10.0;
            }
        }

        // Measure execution time
        double startTime = getTimeInSeconds();

        // Perform Gaussian elimination
        // (Code for Gaussian elimination goes here)

        double endTime = getTimeInSeconds();
        double elapsedTime = endTime - startTime;

        // Store the execution time
        execution_times[run_idx] = elapsedTime;

        // Print the execution time for this run
        printf("Run %d: Execution Time = %.6f seconds\n", run_idx + 1, elapsedTime);
    }

    // Calculate basic statistics on execution times
    double min_time = execution_times[0];
    double max_time = execution_times[0];
    double sum_times = 0.0;

    for (int i = 0; i < NUM_RUNS; i++) {
        sum_times += execution_times[i];
        if (execution_times[i] < min_time) {
            min_time = execution_times[i];
        }
        if (execution_times[i] > max_time) {
            max_time = execution_times[i];
        }
    }

    double mean_time = sum_times / NUM_RUNS;

    // Print summary statistics
    printf("Summary Statistics:\n");
    printf("Mean Execution Time = %.6f seconds\n", mean_time);
    printf("Min Execution Time = %.6f seconds\n", min_time);
    printf("Max Execution Time = %.6f seconds\n", max_time);

    return 0;
}
