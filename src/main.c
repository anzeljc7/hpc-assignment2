#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "lenia.h"

#define N 256
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// Place two orbiums in the world with different angles. (y, x, angle)
// Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

int main()
{
    double start = omp_get_wtime();
    // Run the simulation
    double *world = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
    double stop = omp_get_wtime();
    printf("Execution time: %.3f\n", stop - start);
    free(world);
    return 0;
}