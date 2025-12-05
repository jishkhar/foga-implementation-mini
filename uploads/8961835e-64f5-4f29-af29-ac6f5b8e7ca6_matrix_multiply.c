// #include <stdio.h>
// #include <stdlib.h>

// // --- Configuration ---
// // Increase this value for a longer-running, more intensive benchmark.
// // A size of 512 is a good starting point.
// #define MATRIX_SIZE 512

// // Function to allocate memory for a matrix
// double** create_matrix(int size) {
//     double** matrix = (double**)malloc(size * sizeof(double*));
//     for (int i = 0; i < size; i++) {
//         matrix[i] = (double*)malloc(size * sizeof(double));
//     }
//     return matrix;
// }

// // Function to free the memory of a matrix
// void free_matrix(double** matrix, int size) {
//     for (int i = 0; i < size; i++) {
//         free(matrix[i]);
//     }
//     free(matrix);
// }

// int main() {
//     int n = MATRIX_SIZE;

//     // 1. Allocate memory for three matrices
//     double** A = create_matrix(n);
//     double** B = create_matrix(n);
//     double** C = create_matrix(n);

//     // 2. Initialize matrices A and B with some values
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             A[i][j] = (double)(i + j);
//             B[i][j] = (double)(i - j);
//             C[i][j] = 0.0;
//         }
//     }

//     // 3. Perform matrix multiplication (C = A * B)
//     // This is the computationally intensive part that FOGA will optimize.
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             for (int k = 0; k < n; k++) {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }

//     // 4. Print a single result to prevent dead code elimination
//     // This ensures the compiler must perform the calculation.
//     printf("Result checksum: %f\n", C[0][0]);

//     // 5. Free allocated memory
//     free_matrix(A, n);
//     free_matrix(B, n);
//     free_matrix(C, n);

//     return 0;
// }








// -------------------------------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>

// --- Configuration ---
// Set the matrix size to 5x5 as requested.
#define MATRIX_SIZE 5

/**
 * @brief Allocates memory for a square matrix of doubles.
 *
 * @param size The dimension (n) for an n x n matrix.
 * @return double** Pointer to the allocated matrix.
 */
double** create_matrix(int size) {
    // Allocate space for 'size' rows (pointers to doubles)
    double** matrix = (double**)malloc(size * sizeof(double*));
    if (matrix == NULL) {
        perror("Failed to allocate memory for matrix rows");
        exit(EXIT_FAILURE);
    }
    // Allocate space for 'size' columns for each row
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
        if (matrix[i] == NULL) {
            perror("Failed to allocate memory for matrix columns");
            // Clean up already allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

/**
 * @brief Frees the dynamically allocated memory of a matrix.
 *
 * @param matrix The matrix to free.
 * @param size The dimension of the matrix.
 */
void free_matrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    int n = MATRIX_SIZE;

    printf("Starting %dx%d matrix multiplication...\n", n, n);

    // 1. Allocate memory for three matrices
    double** A = create_matrix(n);
    double** B = create_matrix(n);
    double** C = create_matrix(n);

    // 2. Initialize matrices A and B with integer-based values,
    // and C with zeros.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(i + j); // A[i][j] = i + j
            B[i][j] = (double)(i - j); // B[i][j] = i - j
            C[i][j] = 0.0;
        }
    }

    // 3. Perform matrix multiplication (C = A * B)
    // The triple-nested loop structure (i-j-k) is used here.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Optional: Print the full result matrix C for verification
    printf("\nResult Matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.2f", C[i][j]);
        }
        printf("\n");
    }

    // 4. Print the checksum (C[0][0])
    // The expected result for C[0][0] (sum of k^2 for k=0 to 4) is 30.0
    printf("\nResult checksum (C[0][0]): %f\n", C[0][0]);

    // 5. Free allocated memory
    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}
