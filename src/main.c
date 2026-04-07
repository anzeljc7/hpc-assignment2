#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "lenia.h"

#define NUM_STEPS 100
#define DT 0.1f
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// Koordinate se prilagodijo spodaj glede na N
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, 0, 0}, {0, 0, 180}}; 

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Uporaba: %s <velikost_N>\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);

    // Dinamična postavitev orbiumov glede na podan N
    orbiums[0].row = 0;
    orbiums[0].col = N / 3;
    orbiums[1].row = N / 3;
    orbiums[1].col = 0;

    double start = omp_get_wtime();
    
    float *world = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS);
    
    double stop = omp_get_wtime();
    
    // Bash skripta pri računanju povprečja išče točno to
    printf("Execution time: %.3f\n", stop - start);
    
    free(world);
    return 0;
}