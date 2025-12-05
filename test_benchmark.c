Com#include <stdio.h>

int main() {
    long long sum = 0;
    for (int i = 0; i < 10000000; i++) {
        sum += i;
    }
    printf("Sum: %lld\n", sum);
    return 0;
}
