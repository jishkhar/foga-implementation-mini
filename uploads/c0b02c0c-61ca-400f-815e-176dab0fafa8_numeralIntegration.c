#include <stdio.h>
#include <stdlib.h>

// --- Configuration ---
// Define the number of intervals for numerical integration.
// Increase this value for a more intensive benchmark.
#define NUM_INTERVALS 1000000

// Define PI without using math.h
#define PI 3.14159265358979323846

/**
 * @brief Function to integrate. Modify this function for different benchmarks.
 * Using a polynomial function that's computationally intensive but doesn't require math.h
 *
 * @param x The input value.
 * @return double The function value at x.
 */
double function_to_integrate(double x) {
    // Integrate: x^4 - 3*x^3 + 2*x^2 - x + 5
    // This is computationally intensive without requiring math library
    return x*x*x*x - 3.0*x*x*x + 2.0*x*x - x + 5.0;
}

/**
 * @brief Performs numerical integration using the trapezoidal rule.
 *
 * @param a The lower limit of integration.
 * @param b The upper limit of integration.
 * @param n The number of intervals.
 * @return double The computed integral value.
 */
double trapezoidal_rule(double a, double b, int n) {
    double h = (b - a) / n; // Step size
    double integral = 0.0;

    // Add the first and last terms
    integral += function_to_integrate(a) / 2.0;
    integral += function_to_integrate(b) / 2.0;

    // Add the middle terms
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        integral += function_to_integrate(x);
    }

    integral *= h; // Multiply by step size
    return integral;
}

int main() {
    double a = 0.0; // Lower limit of integration
    double b = PI; // Upper limit of integration (e.g., integrate over [0, π])
    int n = NUM_INTERVALS;

    printf("Starting numerical integration with %d intervals...\n", n);

    // Perform numerical integration
    double result = trapezoidal_rule(a, b, n);

    // Print the result
    printf("Integral of f(x) = x^4 - 3x^3 + 2x^2 - x + 5 from %.2f to %.2f is approximately: %.10f\n", a, b, result);

    return 0;
}