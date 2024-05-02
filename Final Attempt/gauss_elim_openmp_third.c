#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>  // Include OpenMP header

#define MAX_N 1000  // Maximum dimension of the matrix

double getTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

void gaussianElimination(float A[MAX_N][MAX_N + 1], int N) {
    for (int j = 0; j < N - 1; j++) {
        int max_row = j;
        float max_val = A[j][j];
        
        // Find the row with the maximum value in the current column
        for (int i = j + 1; i < N; i++) {
            if (A[i][j] > max_val) {
                max_val = A[i][j];
                max_row = i;
            }
        }
        
        // Swap rows if necessary
        if (max_row != j) {
            for (int k = j; k <= N; k++) {
                float temp = A[j][k];
                A[j][k] = A[max_row][k];
                A[max_row][k] = temp;
            }
        }
        
        // Gaussian elimination
        #pragma omp parallel for default(none) shared(A, N, j) num_threads(omp_get_max_threads())
        for (int i = j + 1; i < N; i++) {
            float factor = A[i][j] / A[j][j];
            for (int k = j; k <= N; k++) {
                A[i][k] -= factor * A[j][k];
            }
        }
    }
}

void backSubstitution(float A[MAX_N][MAX_N + 1], float x[MAX_N], int N) {
    for (int i = N - 1; i >= 0; i--) {
        x[i] = A[i][N];
        for (int j = i + 1; j < N; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}

int main() {
    int N = 1000;  // Matrix dimension
    float A[MAX_N][MAX_N + 1];
    float x[MAX_N];

    srand(time(NULL));

    // Initialize matrix A with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= N; j++) {
            A[i][j] = (float)rand() / RAND_MAX * 10.0;
        }
    }

    // Perform Gaussian elimination
    double startTime = getTimeInSeconds();
    gaussianElimination(A, N);
    double endTime = getTimeInSeconds();
    printf("Gaussian elimination time: %.6f seconds\n", endTime - startTime);

    // Perform back substitution to find the solution
    startTime = getTimeInSeconds();
    backSubstitution(A, x, N);
    endTime = getTimeInSeconds();
    printf("Back substitution time: %.6f seconds\n", endTime - startTime);

    // Print the solution vector
    printf("Solution vector:\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %.6f\n", i, x[i]);
    }

    return 0;
}
